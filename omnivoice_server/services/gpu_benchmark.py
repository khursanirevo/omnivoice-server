"""
Auto-benchmark: find optimal batch size and timeout for the current GPU.

Tests increasing batch sizes, picks the one with the best throughput,
and derives an optimal batch_timeout_ms from actual compute latency.
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omnivoice import OmniVoice

logger = logging.getLogger(__name__)

# Batch sizes to test (skip 16 — known torch.compile recompilation spike)
BATCH_SIZES = [1, 2, 4, 8, 32, 64, 128]
TEXT = "Benchmark test sentence for batch throughput measurement."


def _get_vram_mb() -> int:
    """Get current GPU VRAM usage in MB."""
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024 / 1024)


def _run_batch(model: OmniVoice, batch_size: int, num_step: int) -> tuple[float, int]:
    """Run one batched inference. Returns (elapsed_seconds, num_ok_outputs)."""
    texts = [TEXT] * batch_size
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(text=texts, num_step=num_step)
    elapsed = time.perf_counter() - t0
    nans = sum(1 for t in output if isinstance(t, torch.Tensor) and torch.isnan(t).any().item())
    ok = len(output) - nans
    return elapsed, ok


def _compute_optimal_timeout(single_latency_s: float) -> int:
    """
    Derive optimal batch_timeout_ms from single-request compute latency.

    timeout = 50% of single request latency, clamped to [5, 100] ms.
    - Fast GPU (30ms/request): ~15ms — low latency, still catches bursts
    - Slow GPU (200ms/request): ~100ms — enough to build batches
    """
    timeout_ms = round(single_latency_s * 0.5 * 1000)
    return max(5, min(100, timeout_ms))


def _pick_optimal(results: list[dict]) -> dict:
    """
    Pick optimal batch size: best throughput, with tie-breaker for lower
    latency (smaller batch).

    Instead of stopping at first plateau (which misses real gains at
    higher batch sizes), we pick the actual peak throughput. If two
    batch sizes are within 3% throughput, prefer the smaller one
    (lower per-request latency, less VRAM).
    """
    if not results:
        return {"batch_size": 4, "throughput_req_s": 0.0}

    # Find peak throughput
    best = max(results, key=lambda r: r["throughput_req_s"])
    best_tp = best["throughput_req_s"]

    # Among all results within 3% of peak, pick the smallest batch
    # (lower latency, less VRAM, same throughput)
    near_peak = [r for r in results if r["throughput_req_s"] >= best_tp * 0.97]
    chosen = min(near_peak, key=lambda r: r["batch_size"])

    if chosen["batch_size"] != best["batch_size"]:
        logger.info(
            "Peak throughput at batch=%d (%.1f req/s), "
            "but batch=%d gives 97%%+ of peak (%.1f req/s) with lower latency",
            best["batch_size"], best_tp,
            chosen["batch_size"], chosen["throughput_req_s"],
        )

    return chosen


def find_optimal_batch_size(
    model: OmniVoice,
    num_step: int = 16,
    bench_rounds: int = 3,
) -> dict:
    """
    Benchmark increasing batch sizes and find the optimal configuration.

    Uses the production num_step for accurate results.

    Returns dict with:
        optimal_batch_size: int
        optimal_batch_timeout_ms: int
        throughput_req_s: float
        results: list of {batch_size, latency_s, throughput_req_s, vram_mb}
    """
    logger.info(
        "Starting GPU auto-benchmark (num_step=%d)...", num_step,
    )

    # Warmup for each batch size (trigger compilation)
    warmup_step = min(num_step, 4)
    for bs in [1, 2, 4, 8]:
        try:
            _run_batch(model, bs, warmup_step)
        except Exception as e:
            logger.warning("Warmup batch=%d failed: %s", bs, e)

    # Benchmark each batch size with production num_step
    results = []
    for bs in BATCH_SIZES:
        try:
            timings = []
            total_ok = 0

            for _ in range(bench_rounds):
                elapsed, ok = _run_batch(model, bs, num_step)
                if ok == bs:
                    timings.append(elapsed)
                    total_ok += ok
                else:
                    logger.warning("Batch=%d: %d/%d outputs had NaN", bs, bs - ok, bs)

            if not timings:
                logger.warning("Batch=%d: all attempts had errors, stopping", bs)
                break

            mean_latency = statistics.mean(timings)
            total_wall = sum(timings)
            throughput = total_ok / total_wall
            vram_after = _get_vram_mb()

            entry = {
                "batch_size": bs,
                "latency_s": round(mean_latency, 3),
                "throughput_req_s": round(throughput, 2),
                "vram_mb": vram_after,
            }
            results.append(entry)
            logger.info(
                "  batch=%2d: %.3fs, %.1f req/s, %d MB VRAM",
                bs, mean_latency, throughput, vram_after,
            )

        except Exception as e:
            logger.warning("Batch=%d failed: %s — stopping search", bs, e)
            gc.collect()
            torch.cuda.empty_cache()
            break

    if not results:
        logger.warning("Benchmark failed, defaulting to batch_size=4")
        return {
            "optimal_batch_size": 4,
            "optimal_batch_timeout_ms": 50,
            "optimal_throughput_req_s": 0.0,
            "results": [],
        }

    # Pick optimal: best throughput, prefer smaller batch within 3% of peak
    optimal = _pick_optimal(results)

    # Derive optimal timeout from single-request latency
    single_latency = results[0]["latency_s"]
    timeout_ms = _compute_optimal_timeout(single_latency)

    vram_end = _get_vram_mb()
    summary = {
        "optimal_batch_size": optimal["batch_size"],
        "optimal_batch_timeout_ms": timeout_ms,
        "optimal_throughput_req_s": optimal["throughput_req_s"],
        "single_request_latency_ms": round(single_latency * 1000, 1),
        "vram_used_mb": vram_end,
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        "results": results,
    }

    logger.info(
        "Benchmark complete: batch=%d, timeout=%dms, %.1f req/s "
        "(single=%.1fms), VRAM %d/%d MB",
        summary["optimal_batch_size"],
        summary["optimal_batch_timeout_ms"],
        summary["optimal_throughput_req_s"],
        summary["single_request_latency_ms"],
        summary["vram_used_mb"],
        summary["vram_total_mb"],
    )
    return summary
