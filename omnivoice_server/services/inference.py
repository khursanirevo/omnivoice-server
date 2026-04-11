"""
Runs model.generate() in a thread pool with concurrency limiting and
post-request memory cleanup.

DESIGN NOTE — upstream isolation:
  All kwargs construction for model.generate() is centralised in
  OmniVoiceAdapter._build_kwargs(). When OmniVoice adds / renames params,
  only that one method changes — not SynthesisRequest, not the router.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from ..config import Settings
from .model import ModelService

logger = logging.getLogger(__name__)


@dataclass
class SynthesisRequest:
    text: str
    mode: str  # "auto" | "design" | "clone"
    instruct: str | None = None  # for mode="design"
    ref_audio_path: str | None = None  # tmp path, for mode="clone"
    ref_text: str | None = None  # for mode="clone", optional
    speed: float = 1.0
    num_step: int | None = None  # None → use server default
    # Advanced passthrough — None means "use upstream default"
    guidance_scale: float | None = None
    denoise: bool | None = None
    t_shift: float | None = None
    position_temperature: float | None = None
    class_temperature: float | None = None
    duration: float | None = None  # Fixed output duration in seconds
    language: str | None = None  # Optional language code for multilingual pronunciation


@dataclass
class SynthesisResult:
    tensors: list  # list[torch.Tensor], each (1, T)
    duration_s: float
    latency_s: float


class OmniVoiceAdapter:
    """
    Thin adapter that translates SynthesisRequest → model.generate() kwargs.

    WHY THIS EXISTS:
    OmniVoice.generate() accepts ~10 parameters (num_step, speed, instruct,
    ref_audio, ref_text, guidance_scale, denoise, duration, …). As upstream
    adds / renames parameters, only this class needs to change — not the
    request schema, not the router, not the tests.

    This is the single seam between omnivoice-server and the upstream library.
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._clone_prompt_cache: dict[str, object] = {}
        self._cache_lock = threading.Lock()

    def build_kwargs(self, req: SynthesisRequest, model) -> dict:
        """Return kwargs dict ready to pass to model.generate()."""
        num_step = req.num_step or self._cfg.num_step
        guidance_scale = (
            req.guidance_scale if req.guidance_scale is not None else self._cfg.guidance_scale
        )
        denoise = req.denoise if req.denoise is not None else self._cfg.denoise
        t_shift = req.t_shift if req.t_shift is not None else self._cfg.t_shift
        position_temperature = (
            req.position_temperature
            if req.position_temperature is not None
            else self._cfg.position_temperature
        )
        class_temperature = (
            req.class_temperature
            if req.class_temperature is not None
            else self._cfg.class_temperature
        )

        kwargs: dict = {
            "text": req.text,
            "num_step": num_step,
            "speed": req.speed,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
            "t_shift": t_shift,
            "position_temperature": position_temperature,
            "class_temperature": class_temperature,
        }

        # Add optional duration parameter if provided
        if req.duration is not None:
            kwargs["duration"] = req.duration

        # Add optional language parameter if provided
        if req.language is not None:
            kwargs["language"] = req.language

        if req.mode == "design" and req.instruct:
            kwargs["instruct"] = req.instruct
        elif req.mode == "clone" and req.ref_audio_path:
            prompt = self._get_or_create_clone_prompt(req, model)
            kwargs["voice_clone_prompt"] = prompt

        return kwargs

    def _get_or_create_clone_prompt(self, req: SynthesisRequest, model):
        """Cache voice_clone_prompt keyed by ref audio file content hash."""
        import pathlib

        audio_bytes = pathlib.Path(req.ref_audio_path).read_bytes()
        cache_key = hashlib.sha256(audio_bytes).hexdigest()

        with self._cache_lock:
            if cache_key in self._clone_prompt_cache:
                logger.debug("Reusing cached voice clone prompt")
                return self._clone_prompt_cache[cache_key]

        logger.info("Encoding voice clone prompt (new ref audio, %d bytes)", len(audio_bytes))
        prompt = model.create_voice_clone_prompt(
            req.ref_audio_path,
            ref_text=req.ref_text,
        )

        with self._cache_lock:
            self._clone_prompt_cache[cache_key] = prompt
            # Evict oldest entries if cache grows too large
            if len(self._clone_prompt_cache) > 32:
                oldest_key = next(iter(self._clone_prompt_cache))
                del self._clone_prompt_cache[oldest_key]

        return prompt

    def call(self, req: SynthesisRequest, model) -> list[torch.Tensor]:
        """Call model.generate() and return raw tensors."""
        kwargs = self.build_kwargs(req, model)
        try:
            return model.generate(**kwargs)
        except TypeError as exc:
            # Upstream renamed or removed a param — try graceful fallback
            # by stripping unknown kwargs one-by-one.
            logger.warning(
                f"model.generate() raised TypeError: {exc}. "
                "Attempting fallback with minimal kwargs."
            )
            minimal = {
                "text": kwargs["text"],
                "num_step": kwargs.get("num_step", 16),
            }
            if "instruct" in kwargs:
                minimal["instruct"] = kwargs["instruct"]
            if "voice_clone_prompt" in kwargs:
                minimal["voice_clone_prompt"] = kwargs["voice_clone_prompt"]
            return model.generate(**minimal)


class InferenceService:
    def __init__(
        self,
        model_svc: ModelService,
        cfg: Settings,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._model_svc = model_svc
        self._cfg = cfg
        self._executor = executor
        self._semaphore = (
            asyncio.Semaphore(cfg.max_concurrent) if executor else None
        )
        self._adapter = OmniVoiceAdapter(cfg)

    async def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        if self._executor is not None:
            return await self._synthesize_threaded(req)
        return await self._synthesize_direct(req)

    async def _synthesize_direct(self, req: SynthesisRequest) -> SynthesisResult:
        """Direct inference (multi-worker mode). Each worker handles one request at a time."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, req)

    async def _synthesize_threaded(self, req: SynthesisRequest) -> SynthesisResult:
        """Threaded inference with semaphore (single-worker fallback mode)."""
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        self._executor, self._run_sync, req
                    ),
                    timeout=self._cfg.request_timeout_s,
                )
                return result
            except asyncio.TimeoutError:
                raise

    def _run_sync(self, req: SynthesisRequest) -> SynthesisResult:
        """Blocking inference. Runs in thread pool thread."""
        t0 = time.monotonic()
        model = self._model_svc.model

        try:
            tensors = self._adapter.call(req, model)
        finally:
            if self._executor is not None:
                _cleanup_memory(self._cfg.device)

        duration_s = sum(t.shape[-1] for t in tensors) / 24_000
        latency_s = time.monotonic() - t0

        logger.debug(
            f"Synthesized {duration_s:.2f}s audio in {latency_s:.2f}s "
            f"(RTF={latency_s / duration_s:.3f})"
        )
        return SynthesisResult(
            tensors=tensors,
            duration_s=duration_s,
            latency_s=latency_s,
        )


def _cleanup_memory(device: str) -> None:
    """Post-inference memory cleanup to mitigate potential Torch memory growth."""
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"CUDA cache cleanup failed (non-fatal): {e}")
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception as e:
            logger.debug(f"MPS cache cleanup failed (non-fatal): {e}")
