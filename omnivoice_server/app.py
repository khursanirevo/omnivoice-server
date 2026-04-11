"""
FastAPI application factory.
All shared state lives on app.state — no module-level globals.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .routers import health, models, speech, voices
from .services.inference import InferenceService, SynthesisRequest
from .services.metrics import MetricsService
from .services.model import ModelService
from .services.profiles import ProfileService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Settings = app.state.cfg
    app.state.start_time = time.time()
    app.state.metrics_svc = MetricsService()

    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    app.state.profile_svc = ProfileService(profile_dir=cfg.profile_dir)

    if cfg.workers == 1:
        # Single-worker mode: load model here, use ThreadPoolExecutor
        model_svc = ModelService(cfg)
        app.state.model_svc = model_svc
        executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent)
        app.state.executor = executor
        app.state.inference_svc = InferenceService(model_svc, cfg, executor=executor)

        await model_svc.load()
    else:
        # Multi-worker mode.
        # IMPORTANT SAFETY INVARIANT: This code path only runs inside forked worker
        # processes. The parent process never calls create_app() or runs this lifespan —
        # it goes directly to worker_mgr.monitor() in cli.py. CUDA initialization here
        # is safe because it happens AFTER fork, in an isolated child process.
        model_svc = ModelService(cfg)
        app.state.model_svc = model_svc
        app.state.executor = None
        app.state.inference_svc = InferenceService(model_svc, cfg, executor=None)

        await model_svc.load()

        # Slot 0 worker: measure peak VRAM and write to temp file for parent
        worker_slot = int(os.environ.get("OMNIVOICE_WORKER_SLOT", "-1"))
        if worker_slot == 0 and cfg.device == "cuda":
            import json

            import torch

            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / 1024 / 1024
            free_bytes, total_bytes = torch.cuda.mem_get_info()

            vram_file = "/tmp/omnivoice_vram_measurement.json"

            with open(vram_file, "w") as f:
                json.dump({
                    "peak_vram_mb": round(peak_mb, 2),
                    "total_vram_mb": round(total_bytes / 1024 / 1024, 2),
                }, f)
            logger.info("VRAM measurement written: %.0f MB -> %s", peak_mb, vram_file)

    # Warmup: prime CUDA kernels, memory allocators, and torch JIT caches
    # so the first real request isn't garbage.
    if app.state.model_svc.is_loaded:
        try:
            warmup_text = "Warmup inference for CUDA kernel compilation."
            for i in range(5):
                await app.state.inference_svc.synthesize(
                    SynthesisRequest(text=warmup_text, mode="auto")
                )
            logger.info("Warmup complete (5 inference calls)")
        except Exception:
            logger.debug("Warmup skipped (model not fully initialized)")

    yield

    # Shutdown
    if getattr(app.state, "executor", None):
        app.state.executor.shutdown(wait=False)


def _status_to_code(status_code: int) -> str:
    _map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        422: "validation_error",
        500: "inference_failed",
        503: "model_not_ready",
        504: "timeout",
    }
    return _map.get(status_code, f"http_{status_code}")


def create_app(cfg: Settings) -> FastAPI:
    app = FastAPI(
        title="omnivoice-server",
        description="OpenAI-compatible HTTP server for OmniVoice TTS",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.state.cfg = cfg

    # ── Auth middleware ───────────────────────────────────────────────────────
    if cfg.api_key:

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip auth for health, metrics, and model listing
            if request.url.path in ("/health", "/metrics", "/v1/models"):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {cfg.api_key}":
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Invalid or missing API key"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "detail": exc.errors(),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _status_to_code(exc.status_code),
                    "message": exc.detail,
                }
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(speech.router, prefix="/v1")
    app.include_router(voices.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")
    app.include_router(health.router)

    return app
