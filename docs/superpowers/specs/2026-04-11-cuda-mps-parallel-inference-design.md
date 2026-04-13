# CUDA MPS Parallel Inference Design

**Date:** 2026-04-11
**Status:** Draft
**Scope:** Replace blocking single-process inference with NVIDIA MPS-proxied multi-worker parallel inference

---

## Problem

Current architecture uses a single uvicorn worker with `ThreadPoolExecutor(max_workers=2)` and `asyncio.Semaphore(2)`. All GPU inference shares the default CUDA stream, serializing kernel execution. Concurrent `model.generate()` calls on the same model instance risk internal state corruption. The GIL further limits true parallelism. Per-request `gc.collect()` + `torch.cuda.empty_cache()` creates contention under load.

**Target hardware:** Single NVIDIA GPU (device 0 only)
**Target concurrency:** 4-8 concurrent inference requests

---

## Architecture

### Current (Blocking)

```
Single Process
  ├── ThreadPoolExecutor(max_workers=2)
  │     └── Thread 1: model.generate() ──┐ GPU serialized (default stream)
  │     └── Thread 2: model.generate() ──┘
  └── asyncio.Semaphore(2)
```

### Proposed (MPS-Proxied Parallel)

```
Client Request
      │
      ▼
Parent binds TCP socket (SO_REUSEPORT)
      │
┌─────┼─────┬─────┐
▼     ▼     ▼     ▼
W1    W2    W3    W4      ← N worker processes
│     │     │     │       Each: own CUDA context + own model in VRAM
└─────┼─────┼─────┘
      │
┌─────▼───────────┐
│  MPS Daemon     │     ← NVIDIA CUDA Multi-Process Service
│  (context proxy)│        Shares compute (SM scheduling), NOT memory
└─────┬───────────┘
      │
┌─────▼───────────┐
│  GPU 0          │     ← Concurrent kernel execution
└─────────────────┘
```

### How It Works

1. MPS daemon starts first as GPU context proxy
2. Parent binds TCP socket with `SO_REUSEPORT`, then forks N workers
3. Each worker independently loads the model onto GPU (separate CUDA context, separate VRAM allocation)
4. MPS shares GPU **compute** (SM scheduling) across workers — not memory
5. Each worker runs its own FastAPI + uvicorn (workers=1), accepting on inherited socket fd
6. Each worker is fully isolated — no shared model state, no GIL contention
7. OS kernel distributes incoming connections across workers (round-robin via `SO_REUSEPORT`)

**Key constraint: `fork()` after CUDA context initialization is undefined behavior (NVIDIA).** Workers must NOT inherit a CUDA context from the parent. The parent never initializes CUDA. Each worker loads the model independently after fork.

---

## Component Changes

### New Files

| File | Purpose |
|------|---------|
| `omnivoice_server/mps.py` | MPS daemon lifecycle (start/stop/status) |
| `omnivoice_server/worker_manager.py` | Worker pool spawning, monitoring, crash recovery |

### Modified Files

| File | Change |
|------|--------|
| `omnivoice_server/config.py` | Add `workers`, `mps_enabled`, `mps_active_thread_percentage` settings |
| `omnivoice_server/cli.py` | Add `--workers` flag, orchestrate MPS + worker startup |
| `omnivoice_server/app.py` | Remove ThreadPoolExecutor and Semaphore (no longer needed per-worker); each worker runs lifespan independently |
| `omnivoice_server/services/inference.py` | Simplify to single-threaded inference (concurrency handled by worker count); remove semaphore, thread pool, per-request gc.collect |
| `omnivoice_server/services/profiles.py` | Add `fcntl.flock()` for concurrent profile writes across workers |
| `Dockerfile` | Switch to CUDA base image (`nvidia/cuda` or `pytorch/pytorch`), install CUDA PyTorch, add MPS entrypoint |
| `docker-compose.yml` | MPS device permissions, `deploy.resources.reservations.devices` for GPU |

### Unchanged Files

- `routers/` — each worker runs the same FastAPI app
- `services/metrics.py` — per-worker counters (acceptable: each worker tracks its own)
- `voice_presets.py` — unchanged
- `utils/` — unchanged

---

## Configuration

### New Settings

| Setting | Type | Default | Env Var | Description |
|---------|------|---------|---------|-------------|
| `workers` | `int` | `4` | `OMNIVOICE_WORKERS` | Worker processes (1-16) |
| `mps_enabled` | `str` | `"auto"` | `OMNIVOICE_MPS_ENABLED` | `auto`/`true`/`false` |
| `mps_active_thread_percentage` | `int` | `100` | `OMNIVOICE_MPS_ACTIVE_THREAD_PERCENTAGE` | GPU compute % for MPS (1-100) |

`mps_enabled=auto` logic: enable when `device=cuda` and `workers > 1`.

### Environment Variables for Workers

```
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
CUDA_VISIBLE_DEVICES=0
```

`CUDA_DEVICE_MAX_CONNECTIONS` left at default — no override without empirical justification.

---

## MPS Daemon Management

### Startup (`mps.py`)

```
1. Verify GPU present: torch.cuda.is_available()
2. Check MPS not already running
3. mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
4. Export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
5. Export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
6. Start: nvidia-cuda-mps-control -d
7. Verify: echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control
8. Set thread % if non-default: echo "set_default_active_thread_percentage <val>" | nvidia-cuda-mps-control
```

### Shutdown

```
1. echo "quit" | nvidia-cuda-mps-control
2. Wait for daemon exit (timeout 5s)
3. Log shutdown status
```

### Health Check

Parent process checks MPS daemon every 30s. **MPS daemon death is catastrophic** — all worker CUDA contexts become permanently invalid. Recovery requires killing all workers and restarting from scratch (MPS daemon + all workers). This means up to 30s of failed requests before detection.

---

## Worker Lifecycle

### Startup Sequence

```
1. Parse config (parent process, NO CUDA initialization)
2. If cuda + workers > 1 → start MPS daemon (mps.py)
3. Parent binds TCP socket with SO_REUSEPORT on configured host:port
4. Fork N workers:
   a. Each child independently: initialize CUDA, load model via ModelService
   b. Each child: create own FastAPI app instance with lifespan
   c. Each child: run uvicorn(workers=1) on inherited socket fd
5. Parent monitors children via PID tracking
6. Signal: OMNIVOICE_READY workers=N mps=true
```

**VRAM guard** (runs per-worker, before model load):
```python
# Worker startup, before loading model:
total_vram_mb = torch.cuda.mem_get_info()[1] / 1024 / 1024
free_vram_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
# Assume ~2x model weight size for inference-time activations/buffers
# Workers load sequentially from parent fork, so check available VRAM
estimated_per_worker_mb = ...  # measured empirically at first worker startup
safe_workers = int(free_vram_mb / (estimated_per_worker_mb * 2.0))
```

First worker to start measures actual VRAM usage after load + one inference. Writes result to a shared temp file. Parent reads it and reduces remaining workers if needed.

### Request Routing

`SO_REUSEPORT` on the TCP socket. The Linux kernel distributes incoming connections across all processes that bound the same port (round-robin on Linux 3.9+). No external load balancer needed.

### Crash Recovery

```
Parent process:
  - Track child PIDs
  - SIGCHLD handler detects crashes
  - Fork replacement worker (new process loads model independently)
  - Crash loop detection: >3 crashes in 60s → stop restarting that slot
  - Log all crash/restart events
```

### Graceful Shutdown

```
SIGTERM → Parent:
  1. SIGTERM to all workers
  2. Workers: stop accepting, finish current request, exit
  3. Wait up to shutdown_timeout
  4. SIGKILL any remaining workers
  5. Stop MPS daemon
  6. Exit
```

---

## Error Handling & Fallback

### Fallback Chain

```
workers=4, device=auto
  → CUDA available + MPS started → MPS + N workers (full parallel)
  → CUDA available + MPS failed → single worker with existing ThreadPoolExecutor (safe fallback)
  → No CUDA → single worker, CPU only (existing behavior unchanged)
```

When MPS fails to start, the system falls back to the **current single-worker ThreadPoolExecutor pattern** rather than running multiple workers without MPS (which would cause unmanaged CUDA context contention and Nx VRAM usage).

### Failure Modes

| Scenario | Detection | Response |
|----------|-----------|----------|
| MPS fails to start | Non-zero exit from nvidia-cuda-mps-control | Log warning, fall back to single-worker mode |
| MPS daemon dies mid-flight | Periodic health check (30s) | Kill all workers, restart MPS daemon, restart all workers |
| No NVIDIA GPU | `torch.cuda.is_available() == False` | Skip MPS, single-worker CPU |
| GPU OOM on worker load | CUDA OOM exception in worker | Worker exits with error code, parent reduces worker count and forks replacement |
| Worker crash loop (>3/60s) | Crash counter in parent | Stop restarting that slot, keep remaining workers |
| VRAM too small for N workers | Pre-fork calculation from first worker measurement | Parent reduces workers to fit, warns |

### Profile Storage Concurrency

Workers share the filesystem for voice profiles (single-host deployment):
- Reads: lock-free (atomic file reads)
- Writes: `fcntl.flock()` per-profile on `meta.json`
- No database needed
- **Constraint:** This architecture requires single-host deployment. `fcntl.flock()` does not work across container boundaries.

---

## Docker Changes

### Dockerfile

Switch from CPU-only PyTorch to CUDA-enabled base:

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
# or: FROM pytorch/pytorch:2.3.0-cuda12.4-cudnn9-runtime

# Install CUDA PyTorch (not CPU wheel)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# MPS daemon runs as root; workers run as app user
```

### docker-compose.yml

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
environment:
  - OMNIVOICE_DEVICE=cuda
  - OMNIVOICE_WORKERS=4
  - NVIDIA_VISIBLE_DEVICES=0
```

---

## What Gets Removed

- `ThreadPoolExecutor` for inference (in multi-worker mode; kept as fallback)
- `asyncio.Semaphore` for concurrency gating (in multi-worker mode)
- Per-request `gc.collect()` + `torch.cuda.empty_cache()` (each worker manages own memory)
- Timeout orphan threads (each worker is the concurrency unit)

## What Stays

- All routers unchanged
- Voice presets unchanged
- Audio/text utils unchanged
- Auth middleware unchanged
- Profile CRUD unchanged (add file locking)
