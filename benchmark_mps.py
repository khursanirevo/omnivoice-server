"""Benchmark script for CUDA MPS multi-worker parallel inference.

Measures:
- RTF (Real-Time Factor): inference_time / audio_duration
- TTFA (Time To First Audio): latency from request to first response byte
- Max concurrent throughput: how many parallel requests the server handles

Usage:
    python benchmark_mps.py --url http://localhost:8880 --workers 4 --requests 20
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
import urllib.request

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8880"
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog. This is a test of the text to speech system."
WARMUP_REQUESTS = 3


async def wait_for_server(url: str, timeout: int = 300):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("ready"):
                    logger.info("Server ready: %s", data)
                    return data
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError(f"Server not ready after {timeout}s")


async def single_request(session: aiohttp.ClientSession, url: str, text: str, voice: str = "alloy"):
    """Send a single TTS request and measure RTF, TTFA, and wall time."""
    payload = {"model": "omnivoice", "input": text, "voice": voice, "response_format": "wav"}

    start = time.perf_counter()
    first_byte_time = None

    async with session.post(f"{url}/v1/audio/speech", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            return {"status": resp.status, "error": body, "wall_time_s": time.perf_counter() - start}

        # Read with TTFA measurement
        chunks = []
        async for chunk in resp.content.iter_any():
            if first_byte_time is None:
                first_byte_time = time.perf_counter()
            chunks.append(chunk)

    end = time.perf_counter()
    audio_bytes = b"".join(chunks)

    # Parse WAV duration — use wave module for robustness
    # (torchcodec inserts LIST/INFO chunks before data, so raw offset
    # parsing breaks)
    audio_duration_s = 0
    if len(audio_bytes) > 44:
        import io
        import wave

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    audio_duration_s = frames / rate
        except Exception:
            pass

    wall_time_s = end - start
    ttfa_s = (first_byte_time - start) if first_byte_time else wall_time_s
    rtf = wall_time_s / audio_duration_s if audio_duration_s > 0 else float("inf")

    # Check response headers
    duration_header = resp.headers.get("X-Audio-Duration-S")
    latency_header = resp.headers.get("X-Synthesis-Latency-S")

    return {
        "status": resp.status,
        "wall_time_s": round(wall_time_s, 4),
        "ttfa_s": round(ttfa_s, 4),
        "rtf": round(rtf, 4),
        "audio_duration_s": round(audio_duration_s, 4),
        "audio_bytes": len(audio_bytes),
        "duration_header": duration_header,
        "latency_header": latency_header,
    }


async def warmup(url: str, count: int = WARMUP_REQUESTS):
    """Send warmup requests to prime the model."""
    logger.info("Warming up with %d requests...", count)
    async with aiohttp.ClientSession() as session:
        for i in range(count):
            result = await single_request(session, url, DEFAULT_TEXT)
            logger.info("  Warmup %d: status=%s wall=%.2fs rtf=%.3f",
                        i + 1, result.get("status"), result.get("wall_time_s", 0), result.get("rtf", 0))
    logger.info("Warmup complete")


async def benchmark_sequential(url: str, num_requests: int):
    """Benchmark sequential (1 at a time) requests."""
    logger.info("=== Sequential Benchmark (%d requests) ===", num_requests)
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            r = await single_request(session, url, DEFAULT_TEXT)
            results.append(r)
            logger.info("  [%d/%d] wall=%.2fs rtf=%.3f ttfa=%.3fs",
                        i + 1, num_requests, r.get("wall_time_s", 0),
                        r.get("rtf", 0), r.get("ttfa_s", 0))
    return results


async def benchmark_concurrent(url: str, num_concurrent: int, num_batches: int):
    """Benchmark concurrent requests.

    Args:
        num_concurrent: How many requests in parallel per batch
        num_batches: How many batches to run
    """
    logger.info("=== Concurrent Benchmark (%d concurrent x %d batches) ===",
                num_concurrent, num_batches)
    all_results = []

    async with aiohttp.ClientSession() as session:
        for batch in range(num_batches):
            batch_start = time.perf_counter()
            tasks = [single_request(session, url, DEFAULT_TEXT) for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            batch_time = time.perf_counter() - batch_start

            successful = [r for r in results if r.get("status") == 200]
            failed = [r for r in results if r.get("status") != 200]

            all_results.extend(results)

            rtf_values = [r["rtf"] for r in successful if "rtf" in r]
            ttfa_values = [r["ttfa_s"] for r in successful if "ttfa_s" in r]
            wall_values = [r["wall_time_s"] for r in successful if "wall_time_s" in r]

            logger.info("  Batch %d/%d: %d/%d success, batch_wall=%.2fs",
                        batch + 1, num_batches, len(successful), num_concurrent, batch_time)
            if rtf_values:
                logger.info("    RTF: mean=%.3f median=%.3f min=%.3f max=%.3f",
                            statistics.mean(rtf_values), statistics.median(rtf_values),
                            min(rtf_values), max(rtf_values))
            if ttfa_values:
                logger.info("    TTFA: mean=%.3fs median=%.3fs min=%.3fs max=%.3fs",
                            statistics.mean(ttfa_values), statistics.median(ttfa_values),
                            min(ttfa_values), max(ttfa_values))
            if wall_values:
                throughput = len(successful) / batch_time
                logger.info("    Throughput: %.2f req/s (batch total: %.2fs)", throughput, batch_time)
            if failed:
                for f in failed:
                    logger.warning("    FAILED: status=%s error=%s", f.get("status"), str(f.get("error", ""))[:100])

    return all_results


def print_summary(label: str, results: list):
    """Print summary statistics for a set of results."""
    successful = [r for r in results if r.get("status") == 200]
    if not successful:
        logger.warning("No successful results for %s", label)
        return

    rtf_values = [r["rtf"] for r in successful]
    ttfa_values = [r["ttfa_s"] for r in successful]
    wall_values = [r["wall_time_s"] for r in successful]
    durations = [r["audio_duration_s"] for r in successful]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Successful: {len(successful)}/{len(results)}")
    print(f"  Audio duration: {statistics.mean(durations):.2f}s (avg)")
    print(f"")
    print(f"  RTF (lower=better, <1.0=faster than real-time):")
    print(f"    Mean:   {statistics.mean(rtf_values):.3f}")
    print(f"    Median: {statistics.median(rtf_values):.3f}")
    print(f"    Min:    {min(rtf_values):.3f}")
    print(f"    Max:    {max(rtf_values):.3f}")
    if len(rtf_values) > 2:
        print(f"    Stdev:  {statistics.stdev(rtf_values):.3f}")
        sorted_rtf = sorted(rtf_values)
        p95_idx = int(len(sorted_rtf) * 0.95)
        print(f"    P95:    {sorted_rtf[p95_idx]:.3f}")
    print(f"")
    print(f"  TTFA (Time To First Audio):")
    print(f"    Mean:   {statistics.mean(ttfa_values):.3f}s")
    print(f"    Median: {statistics.median(ttfa_values):.3f}s")
    print(f"    Min:    {min(ttfa_values):.3f}s")
    print(f"    Max:    {max(ttfa_values):.3f}s")
    print(f"")
    print(f"  Wall time per request:")
    print(f"    Mean:   {statistics.mean(wall_values):.2f}s")
    print(f"    Median: {statistics.median(wall_values):.2f}s")
    print(f"{'='*60}\n")


async def find_max_concurrent(url: str, start: int = 1, end: int = 16):
    """Binary search to find max concurrent requests the server can handle."""
    logger.info("=== Finding Max Concurrent (binary search %d-%d) ===", start, end)

    max_ok = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        # Test powers of 2 first, then refine
        for n in [1, 2, 4, 8, 12, 16]:
            if n > end:
                break
            logger.info("  Testing %d concurrent...", n)
            tasks = [single_request(session, url, DEFAULT_TEXT) for _ in range(n)]
            batch_start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            batch_time = time.perf_counter() - batch_start

            successful = [r for r in results if r.get("status") == 200]
            failed = [r for r in results if r.get("status") != 200]
            rate = len(successful) / batch_time if batch_time > 0 else 0

            logger.info("    %d/%d success in %.2fs (%.2f req/s)",
                        len(successful), n, batch_time, rate)

            if len(successful) == n:
                max_ok = n
                max_rate = rate
            else:
                logger.info("    FAILED at %d concurrent (%d failures) - stopping",
                            n, len(failed))
                for f in failed[:3]:
                    logger.info("      Error: status=%s", f.get("status"))
                break

    logger.info("Max concurrent: %d (at %.2f req/s)", max_ok, max_rate if max_ok > 0 else 0)
    return max_ok


async def main():
    parser = argparse.ArgumentParser(description="Benchmark omnivoice-server MPS")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server URL")
    parser.add_argument("--warmup", type=int, default=WARMUP_REQUESTS, help="Warmup requests")
    parser.add_argument("--sequential", type=int, default=10, help="Sequential test count")
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent requests per batch")
    parser.add_argument("--batches", type=int, default=5, help="Concurrent batches")
    parser.add_argument("--find-max", action="store_true", help="Find max concurrent")
    args = parser.parse_args()

    # Wait for server
    health = await wait_for_server(args.url)
    print(f"\nServer: {health}\n")

    # Warmup
    await warmup(args.url, args.warmup)

    # Sequential benchmark
    seq_results = await benchmark_sequential(args.url, args.sequential)
    print_summary("Sequential Benchmark", seq_results)

    # Concurrent benchmark
    conc_results = await benchmark_concurrent(args.url, args.concurrent, args.batches)
    print_summary(f"Concurrent Benchmark ({args.concurrent} parallel)", conc_results)

    # Find max concurrent
    if args.find_max:
        max_c = await find_max_concurrent(args.url)
        print(f"\nMax concurrent: {max_c}")


if __name__ == "__main__":
    asyncio.run(main())
