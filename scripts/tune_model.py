"""AITune AOT tuning script for OmniVoice LLM backbone.

Tunes the Qwen3-0.6B LLM backbone using AITune backends
(TensorRT, TorchInductor) and benchmarks results.

Usage:
    # Tune with TorchInductor (recommended - works with dynamic shapes)
    CUDA_VISIBLE_DEVICES=3 python scripts/tune_model.py

    # Tune with TensorRT (may fail with dynamic shapes)
    CUDA_VISIBLE_DEVICES=3 python scripts/tune_model.py --backend tensorrt

    # Just benchmark (no tuning)
    CUDA_VISIBLE_DEVICES=3 python scripts/tune_model.py --benchmark-only
"""

import argparse
import gc
import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(device, model_id):
    """Load OmniVoice model."""
    from omnivoice import OmniVoice

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        try:
            model = OmniVoice.from_pretrained(
                model_id,
                device_map=f"{device}:0" if device == "cuda" else device,
                dtype=dtype,
            )
            test = model.generate(text="test", num_step=4)
            if any(torch.isnan(t).any() for t in test):
                del model
                gc.collect()
                continue
            return model
        except Exception as e:
            logger.warning("Failed with dtype=%s: %s", dtype, e)
    raise RuntimeError("Could not load model")


def tune_llm_aitune(model, backend_name="first_wins", output_path="tuned_llm.ait"):
    """Tune the LLM backbone using AITune."""
    import aitune.torch as ait
    from aitune.torch.backend import (
        TensorRTBackend,
        TorchEagerBackend,
        TorchInductorBackend,
    )

    backends = {
        "tensorrt": [TensorRTBackend()],
        "inductor": [TorchInductorBackend()],
        "eager": [TorchEagerBackend()],
        "first_wins": [
            TensorRTBackend(),
            TorchInductorBackend(),
            TorchEagerBackend(),
        ],
    }

    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}. Use: {list(backends.keys())}")

    strategy = ait.FirstWinsStrategy(backends=backends[backend_name])

    logger.info("Wrapping LLM backbone with AITune Module")
    model.llm = ait.Module(model.llm, name="omnivoice-llm", strategy=strategy)

    # Create sample inputs for the raw LLM forward pass.
    # The LLM takes inputs_embeds (B, T, H) and attention_mask.
    logger.info("Creating sample input data for tuning...")

    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    config = OmniVoiceGenerationConfig(num_step=2)
    task = model._preprocess_all(text="Hello world test", preprocess_prompt=True)
    inputs = model._prepare_inference_inputs(
        text=task.texts[0],
        num_target_tokens=task.target_lens[0],
        lang=task.langs[0],
        denoise=config.denoise,
    )

    # Compute the actual embeddings from the OmniVoice embedding layers
    B = 2  # conditional + unconditional
    c_len = inputs["input_ids"].shape[2]

    with torch.no_grad():
        inputs_embeds = model._prepare_embed_inputs(
            inputs["input_ids"].repeat(B, 1, 1),
            inputs["audio_mask"].repeat(B, 1),
        )
        attention_mask = torch.ones(
            B, 1, c_len, c_len, dtype=torch.bool, device=model.device
        )

    logger.info("Sample input shapes:")
    logger.info("  inputs_embeds: %s (dtype=%s)", inputs_embeds.shape, inputs_embeds.dtype)
    logger.info("  attention_mask: %s", attention_mask.shape)

    # Define inference function that calls the LLM directly
    def llm_forward(inputs_embeds, attention_mask):
        return model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

    # Tune
    logger.info("Starting AITune tuning with backend=%s ...", backend_name)
    t0 = time.perf_counter()

    try:
        ait.tune(
            func=llm_forward,
            dataset=[{
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }],
            batch_sizes=[1, 2],
            device=model.device,
        )
        elapsed = time.perf_counter() - t0
        logger.info("AITune tuning completed in %.1fs", elapsed)

        # Save
        ait.save(model.llm, output_path)
        logger.info("Tuned model saved to %s", output_path)

        return model

    except Exception as e:
        logger.error("AITune tuning failed: %s", e)
        import traceback
        traceback.print_exc()
        return None


def benchmark_tuned(model, num_runs=5):
    """Quick benchmark of tuned model."""
    logger.info("Benchmarking tuned model (%d runs)...", num_runs)

    # Warmup
    for _ in range(2):
        model.generate(text="Warmup text for benchmarking.", num_step=16)

    timings = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        output = model.generate(text="The quick brown fox jumps over the lazy dog.", num_step=16)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        dur = output[0].shape[-1] / 24000
        rtf = elapsed / dur if dur > 0 else float("inf")
        timings.append(elapsed)
        logger.info("  [%d/%d] wall=%.3fs dur=%.2fs rtf=%.3f", i + 1, num_runs, elapsed, dur, rtf)

    import statistics
    avg = statistics.mean(timings)
    logger.info("Average wall time: %.3fs", avg)
    return avg


def main():
    parser = argparse.ArgumentParser(description="AITune AOT tuning for OmniVoice")
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="first_wins",
                        choices=["tensorrt", "inductor", "eager", "first_wins"])
    parser.add_argument("--output", default="tuned_llm.ait")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Load existing .ait checkpoint")
    args = parser.parse_args()

    model = load_model(args.device, args.model_id)

    if args.checkpoint:
        import aitune.torch as ait
        logger.info("Loading AITune checkpoint from %s", args.checkpoint)
        model.llm = ait.Module(model.llm, name="omnivoice-llm")
        model.llm = ait.load(model.llm, args.checkpoint)
        logger.info("Checkpoint loaded")

    if not args.benchmark_only:
        model = tune_llm_aitune(model, args.backend, args.output)
        if model is None:
            logger.error("Tuning failed, exiting")
            return

    benchmark_tuned(model)


if __name__ == "__main__":
    main()
