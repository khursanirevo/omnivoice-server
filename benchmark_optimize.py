"""Benchmark script for model optimization.

Measures OmniVoice inference performance at different optimization levels:
- Per-step LLM forward timing
- Total generate() wall time
- Audio duration / RTF
- Memory usage

Usage:
    # Standalone (loads model directly, no server needed)
    python benchmark_optimize.py --device cuda --num-step 32 16 8

    # With torch.compile
    python benchmark_optimize.py --device cuda --compile max-autotune

    # With TorchAO quantization
    python benchmark_optimize.py --device cuda --quantization fp8wo

    # Against running server
    python benchmark_optimize.py --url http://localhost:8880 --num-step 32
"""

import argparse
import gc
import io
import json
import logging
import statistics
import time
import wave

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog. This is a test of the text to speech system."
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
SAMPLE_RATE = 24_000


def load_model(device: str, model_id: str, cache_dir: str | None = None):
    """Load OmniVoice model."""
    from omnivoice import OmniVoice

    kwargs = {
        "device_map": f"{device}:0" if device == "cuda" else device,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Try dtype candidates
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        try:
            kwargs["dtype"] = dtype
            logger.info("Loading model with dtype=%s ...", dtype)
            model = OmniVoice.from_pretrained(model_id, **kwargs)
            test = model.generate(text="test", num_step=4)
            if any(torch.isnan(t).any() for t in test):
                logger.warning("dtype=%s produced NaN, trying next", dtype)
                del model
                gc.collect()
                continue
            logger.info("Model loaded with dtype=%s", dtype)
            return model
        except Exception as e:
            logger.warning("Failed with dtype=%s: %s", dtype, e)
            continue

    raise RuntimeError("Could not load model with any dtype")


def apply_compile(model, compile_mode: str, llm_only: bool = True):
    """Apply torch.compile to model."""
    if compile_mode == "none":
        return model

    target = model.llm if llm_only else model
    logger.info("Applying torch.compile(mode=%s) to %s", compile_mode, "LLM backbone" if llm_only else "full model")
    compiled = torch.compile(target, mode=compile_mode)
    if llm_only:
        model.llm = compiled
    else:
        model = compiled
    return model


def apply_quantization(model, quantization: str):
    """Apply TorchAO quantization to model."""
    if quantization == "none":
        return model

    from torchao.quantization import quantize_

    configs = {
        "fp8wo": ("Float8WeightOnlyConfig", {}),
        "fp8dq": ("Float8DynamicActivationFloat8WeightConfig", {}),
        "int8wo": ("Int8WeightOnlyConfig", {}),
        "int8dq": ("Int8DynamicActivationInt8WeightConfig", {}),
    }

    if quantization not in configs:
        raise ValueError(f"Unknown quantization: {quantization}. Use: {list(configs.keys())}")

    import torchao.quantization as tq

    config_cls_name, config_kwargs = configs[quantization]
    config_cls = getattr(tq, config_cls_name)
    config = config_cls(**config_kwargs)

    logger.info("Applying TorchAO quantization=%s to LLM backbone", quantization)
    quantize_(model.llm, config=config)
    logger.info("Quantization applied")
    return model


def benchmark_standalone(args, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """Run standalone benchmark (no server)."""
    model = load_model(args.device, args.model_id, args.cache_dir)

    # Apply optimizations
    if args.quantization != "none":
        model = apply_quantization(model, args.quantization)

    if args.compile != "none":
        model = apply_compile(model, args.compile)

    results = {}

    for num_step in args.num_step:
        label = f"num_step={num_step}"
        if args.compile != "none":
            label += f" compile={args.compile}"
        if args.quantization != "none":
            label += f" quant={args.quantization}"

        logger.info("=" * 60)
        logger.info("Benchmarking: %s", label)
        logger.info("=" * 60)

        # Warmup
        logger.info("Warming up (%d runs)...", warmup)
        for i in range(warmup):
            t0 = time.perf_counter()
            _ = model.generate(text=DEFAULT_TEXT, num_step=num_step)
            elapsed = time.perf_counter() - t0
            logger.info("  Warmup %d: %.2fs", i + 1, elapsed)
            if args.device == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        timings = []
        audio_durations = []

        logger.info("Benchmarking (%d runs)...", runs)
        for i in range(runs):
            torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None

            t0 = time.perf_counter()
            output = model.generate(text=DEFAULT_TEXT, num_step=num_step)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            # Get audio duration
            tensor = output[0]
            audio_dur = tensor.shape[-1] / SAMPLE_RATE
            audio_durations.append(audio_dur)

            rtf = elapsed / audio_dur if audio_dur > 0 else float("inf")
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else 0
            logger.info(
                "  [%2d/%d] wall=%.3fs dur=%.2fs rtf=%.3f peak_vram=%.0fMB",
                i + 1, runs, elapsed, audio_dur, rtf, peak_mem,
            )

        avg_audio_dur = statistics.mean(audio_durations)
        avg_wall = statistics.mean(timings)
        avg_rtf = avg_wall / avg_audio_dur

        result = {
            "label": label,
            "num_step": num_step,
            "compile": args.compile,
            "quantization": args.quantization,
            "runs": runs,
            "wall_time": {
                "mean": round(avg_wall, 4),
                "median": round(statistics.median(timings), 4),
                "min": round(min(timings), 4),
                "max": round(max(timings), 4),
                "stdev": round(statistics.stdev(timings), 4) if len(timings) > 1 else 0,
            },
            "audio_duration": round(avg_audio_dur, 2),
            "rtf": {
                "mean": round(avg_rtf, 4),
                "median": round(statistics.median([t / d for t, d in zip(timings, audio_durations)]), 4),
                "min": round(min(t / d for t, d in zip(timings, audio_durations) if d > 0), 4),
                "max": round(max(t / d for t, d in zip(timings, audio_durations) if d > 0), 4),
            },
        }

        if args.device == "cuda":
            result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 0)

        results[label] = result
        _print_result(result)

    # Save results
    output_file = args.output or "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_file)

    # Print comparison table
    _print_comparison(results)


def benchmark_server(args, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """Run benchmark against running server."""
    import aiohttp
    import asyncio

    async def single_request(session, url, num_step):
        payload = {
            "model": "omnivoice",
            "input": DEFAULT_TEXT,
            "voice": "alloy",
            "response_format": "wav",
        }
        # Note: num_step is server-side config, can't change per-request easily
        t0 = time.perf_counter()
        async with session.post(f"{url}/v1/audio/speech", json=payload) as resp:
            data = await resp.read()
        wall = time.perf_counter() - t0

        dur = 0
        if len(data) > 44 and data[:4] == b"RIFF":
            with wave.open(io.BytesIO(data), "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()

        return {
            "wall": round(wall, 4),
            "duration": round(dur, 2),
            "rtf": round(wall / dur, 4) if dur > 0 else float("inf"),
            "bytes": len(data),
            "status": resp.status,
        }

    async def run():
        url = args.url
        logger.info("Benchmarking server at %s", url)

        for num_step in args.num_step:
            label = f"server num_step={num_step}"
            logger.info("=" * 60)
            logger.info("Benchmarking: %s (%d runs)", label, runs)
            logger.info("=" * 60)

            async with aiohttp.ClientSession() as session:
                # Warmup
                for i in range(warmup):
                    r = await single_request(session, url, num_step)
                    logger.info("  Warmup %d: %.2fs rtf=%.3f", i + 1, r["wall"], r["rtf"])

                # Benchmark
                results_list = []
                for i in range(runs):
                    r = await single_request(session, url, num_step)
                    results_list.append(r)
                    logger.info(
                        "  [%2d/%d] wall=%.3fs dur=%.2fs rtf=%.3f",
                        i + 1, runs, r["wall"], r["duration"], r["rtf"],
                    )

                walls = [r["wall"] for r in results_list]
                durs = [r["duration"] for r in results_list]
                rtfs = [r["rtf"] for r in results_list]

                result = {
                    "label": label,
                    "runs": runs,
                    "wall_time": {
                        "mean": round(statistics.mean(walls), 4),
                        "median": round(statistics.median(walls), 4),
                        "min": round(min(walls), 4),
                        "max": round(max(walls), 4),
                    },
                    "rtf": {
                        "mean": round(statistics.mean(rtfs), 4),
                        "median": round(statistics.median(rtfs), 4),
                        "min": round(min(rtfs), 4),
                        "max": round(max(rtfs), 4),
                    },
                }
                _print_result(result)

    asyncio.run(run())


def _print_result(result):
    """Print a single result block."""
    print(f"\n{'=' * 60}")
    print(f"  {result['label']}")
    print(f"{'=' * 60}")
    wt = result["wall_time"]
    print(f"  Wall time:  mean={wt['mean']:.3f}s  median={wt['median']:.3f}s  "
          f"min={wt['min']:.3f}s  max={wt['max']:.3f}s")
    if "rtf" in result and isinstance(result["rtf"], dict):
        rtf = result["rtf"]
        print(f"  RTF:        mean={rtf['mean']:.3f}  median={rtf['median']:.3f}  "
              f"min={rtf['min']:.3f}  max={rtf['max']:.3f}")
    if "peak_vram_mb" in result:
        print(f"  Peak VRAM:  {result['peak_vram_mb']:.0f} MB")
    if "audio_duration" in result:
        print(f"  Audio dur:  {result['audio_duration']:.2f}s (avg)")
    print(f"{'=' * 60}\n")


def _print_comparison(results):
    """Print comparison table across all configs."""
    if len(results) < 2:
        return

    print(f"\n{'=' * 80}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"  {'Config':<45} {'Wall(s)':>8} {'RTF':>8} {'Speedup':>8}")
    print(f"  {'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8}")

    # Find baseline (first result)
    baseline_wall = list(results.values())[0]["wall_time"]["mean"]

    for label, r in results.items():
        wall = r["wall_time"]["mean"]
        rtf = r["rtf"]["mean"] if isinstance(r.get("rtf"), dict) else r.get("rtf", 0)
        speedup = baseline_wall / wall if wall > 0 else 0
        print(f"  {label:<45} {wall:>8.3f} {rtf:>8.3f} {speedup:>7.2f}x")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OmniVoice optimization")
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice", help="Model ID or path")
    parser.add_argument("--cache-dir", default=None, help="Model cache directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-step", nargs="+", type=int, default=[32, 16, 8],
                        help="num_step values to benchmark")
    parser.add_argument("--compile", default="none",
                        choices=["none", "default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--quantization", default="none",
                        choices=["none", "fp8wo", "fp8dq", "int8wo", "int8dq"],
                        help="TorchAO quantization")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS, help="Benchmark runs")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS, help="Warmup runs")

    # Server mode
    parser.add_argument("--url", default=None, help="Server URL (benchmark against server)")

    args = parser.parse_args()

    if args.url:
        benchmark_server(args, warmup=args.warmup, runs=args.runs)
    else:
        benchmark_standalone(args, warmup=args.warmup, runs=args.runs)


if __name__ == "__main__":
    main()
