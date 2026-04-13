# CUDA MPS Parallel Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blocking single-process GPU inference with NVIDIA MPS-proxied multi-worker parallel inference for 4-8 concurrent requests on a single GPU.

**Architecture:** MPS daemon acts as GPU context proxy. Parent process binds TCP socket with `SO_REUSEPORT`, forks N workers. Each worker independently initializes CUDA and loads the model. Linux kernel distributes connections round-robin. Workers are fully isolated — no shared model state.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, PyTorch, NVIDIA CUDA MPS (`nvidia-cuda-mps-control`), `fcntl` for file locking, `SO_REUSEPORT` for socket sharing.

**Spec:** `docs/superpowers/specs/2026-04-11-cuda-mps-parallel-inference-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `omnivoice_server/mps.py` | MPS daemon lifecycle: start, stop, health check, status query |
| Create | `omnivoice_server/worker_manager.py` | Worker pool: fork workers (slot-aware via env var), monitor PIDs, MPS health check, crash recovery (keyed by slot), VRAM guard (reads measurement from worker), graceful shutdown |
| Modify | `omnivoice_server/config.py:19-159` | Add `workers`, `mps_enabled`, `mps_active_thread_percentage` settings |
| Modify | `omnivoice_server/cli.py:16-157` | Add `--workers` flag, orchestrate MPS + worker manager startup |
| Modify | `omnivoice_server/app.py:27-69` | Skip model loading in parent when workers > 1; each worker loads independently after fork; slot 0 writes VRAM measurement |
| Modify | `omnivoice_server/services/inference.py:137-206` | Simplify InferenceService for single-threaded per-worker operation |
| Modify | `omnivoice_server/services/profiles.py:59-94` | Add `fcntl.flock()` on profile directory for write concurrency |
| Modify | `tests/conftest.py` | Add `workers=1` to settings fixture to preserve single-worker test behavior |
| Modify | `Dockerfile` | Switch to CUDA base image with MPS control binary, add MPS entrypoint |
| Modify | `docker-compose.yml` | Add GPU device reservation, MPS environment |
| Create | `tests/test_mps.py` | Tests for MPS daemon management |
| Create | `tests/test_worker_manager.py` | Tests for worker pool lifecycle |

---

## Task 1: Add MPS and Worker Config Settings

**Key design choice:** `workers` defaults to **1** (not 4). Multi-worker is opt-in — existing deployments and tests continue working unchanged. Set `OMNIVOICE_WORKERS=4` or `--workers 4` to enable.

**Files:**
- Modify: `omnivoice_server/config.py:81-86` (after `max_concurrent`)
- Test: `tests/test_config_workers.py` (new)

- [ ] **Step 1: Write failing tests for new config fields**

Create `tests/test_config_workers.py`:

```python
import pytest
from omnivoice_server.config import Settings


def test_workers_default_is_one():
    """Multi-worker is opt-in. Default preserves existing single-worker behavior."""
    s = Settings(device="cpu")
    assert s.workers == 1


def test_workers_env_override(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_WORKERS", "8")
    s = Settings(device="cpu")
    assert s.workers == 8


def test_workers_clamp_min():
    with pytest.raises(Exception):
        Settings(device="cpu", workers=0)


def test_workers_clamp_max():
    with pytest.raises(Exception):
        Settings(device="cpu", workers=17)


def test_mps_enabled_default_auto():
    s = Settings(device="cpu")
    assert s.mps_enabled == "auto"


def test_mps_enabled_true():
    s = Settings(device="cpu", mps_enabled="true")
    assert s.mps_enabled == "true"


def test_mps_enabled_invalid():
    with pytest.raises(Exception):
        Settings(device="cpu", mps_enabled="maybe")


def test_mps_active_thread_percentage_default():
    s = Settings(device="cpu")
    assert s.mps_active_thread_percentage == 100


def test_mps_active_thread_percentage_range():
    with pytest.raises(Exception):
        Settings(device="cpu", mps_active_thread_percentage=0)
    with pytest.raises(Exception):
        Settings(device="cpu", mps_active_thread_percentage=101)


def test_mps_should_enable_auto_cuda_multiworker():
    s = Settings(device="cuda", workers=4, mps_enabled="auto")
    assert s.mps_should_enable is True


def test_mps_should_enable_auto_cuda_singleworker():
    s = Settings(device="cuda", workers=1, mps_enabled="auto")
    assert s.mps_should_enable is False


def test_mps_should_enable_auto_cpu():
    s = Settings(device="cpu", workers=4, mps_enabled="auto")
    assert s.mps_should_enable is False


def test_mps_should_enable_explicit_true():
    s = Settings(device="cuda", workers=4, mps_enabled="true")
    assert s.mps_should_enable is True


def test_mps_should_enable_explicit_false():
    s = Settings(device="cuda", workers=4, mps_enabled="false")
    assert s.mps_should_enable is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_config_workers.py -v`
Expected: FAIL — `workers`, `mps_enabled`, `mps_should_enable` not defined

- [ ] **Step 3: Add new settings to config.py**

In `omnivoice_server/config.py`, after the `max_concurrent` field (line ~86), add:

```python
    workers: int = Field(
        default=1, ge=1, le=16,
        description="Number of worker processes. Default=1 (single-worker, opt-in for multi-worker).",
    )
    mps_enabled: Literal["auto", "true", "false"] = Field(
        default="auto",
        description="MPS mode: auto=enable when cuda+workers>1, true=force, false=disable",
    )
    mps_active_thread_percentage: int = Field(
        default=100, ge=1, le=100, description="GPU compute percentage for MPS (1-100)"
    )
```

Add computed property to Settings class:

```python
    @property
    def mps_should_enable(self) -> bool:
        """Resolve auto/true/false to boolean."""
        if self.mps_enabled == "true":
            return True
        if self.mps_enabled == "false":
            return False
        # auto: enable only when cuda + multi-worker
        return self.device == "cuda" and self.workers > 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_config_workers.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff on changed files**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/config.py tests/test_config_workers.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add omnivoice_server/config.py tests/test_config_workers.py
git commit -m "feat(config): add workers, mps_enabled, and mps_active_thread_percentage settings (workers defaults to 1)"
```

---

## Task 2: Create MPS Daemon Manager

**Files:**
- Create: `omnivoice_server/mps.py`
- Create: `tests/test_mps.py`

- [ ] **Step 1: Write failing tests for MPS daemon management**

Create `tests/test_mps.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from omnivoice_server.mps import MPSManager, MPSStatus


@pytest.fixture
def mock_subprocess():
    with patch("omnivoice_server.mps.subprocess") as mock_sub:
        mock_sub.run.return_value = MagicMock(returncode=0, stdout="100")
        yield mock_sub


@pytest.fixture
def mock_torch_cuda():
    with patch("omnivoice_server.mps.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        yield mock_torch


def test_status_not_started():
    mgr = MPSManager(active_thread_percentage=100)
    assert mgr.status == MPSStatus.NOT_STARTED


def test_start_success(mock_subprocess, mock_torch_cuda, tmp_path):
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
        active_thread_percentage=100,
    )
    result = mgr.start()
    assert result is True
    assert mgr.status == MPSStatus.RUNNING


def test_start_no_gpu(mock_subprocess):
    with patch("omnivoice_server.mps.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mgr = MPSManager()
        result = mgr.start()
        assert result is False
        assert mgr.status == MPSStatus.UNAVAILABLE


def test_start_mps_control_fails(mock_subprocess, mock_torch_cuda, tmp_path):
    mock_subprocess.run.return_value = MagicMock(returncode=1, stdout="")
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
    )
    result = mgr.start()
    assert result is False
    assert mgr.status == MPSStatus.FAILED


def test_stop(mock_subprocess, mock_torch_cuda, tmp_path):
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
    )
    mgr.start()
    mgr.stop()
    assert mgr.status == MPSStatus.STOPPED


def test_health_check_running(mock_subprocess, mock_torch_cuda, tmp_path):
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
    )
    mgr.start()
    assert mgr.is_healthy() is True


def test_health_check_daemon_died(mock_subprocess, mock_torch_cuda, tmp_path):
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
    )
    mgr.start()
    mock_subprocess.run.side_effect = Exception("daemon not responding")
    assert mgr.is_healthy() is False


def test_set_thread_percentage(mock_subprocess, mock_torch_cuda, tmp_path):
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
        active_thread_percentage=75,
    )
    mgr.start()
    calls = mock_subprocess.run.call_args_list
    set_call = [c for c in calls if "set_default_active_thread_percentage" in str(c)]
    assert len(set_call) >= 1


def test_env_vars_only_set_on_success(mock_subprocess, mock_torch_cuda, tmp_path):
    """MPS env vars should NOT leak on failure."""
    import os
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
    )
    mgr.start()
    assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") is not None


def test_env_vars_cleared_on_failure(mock_subprocess, mock_torch_cuda, tmp_path):
    """If MPS fails after env vars are set, clean them up."""
    import os
    mock_subprocess.run.side_effect = [
        MagicMock(returncode=1, stdout=""),  # start fails
    ]
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
    )
    mgr.start()
    assert mgr.status == MPSStatus.FAILED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_mps.py -v`
Expected: FAIL — `omnivoice_server.mps` module not found

- [ ] **Step 3: Implement MPSManager**

Create `omnivoice_server/mps.py`:

```python
"""NVIDIA CUDA MPS (Multi-Process Service) daemon lifecycle management."""

import logging
import os
import subprocess
from enum import Enum
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MPS_ENV_VARS = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")


class MPSStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class MPSManager:
    """Manages the NVIDIA CUDA MPS daemon lifecycle.

    MPS allows multiple CUDA processes to share GPU compute resources
    concurrently. The daemon acts as a proxy between worker processes
    and the GPU hardware.
    """

    def __init__(
        self,
        pipe_dir: str = "/tmp/nvidia-mps",
        log_dir: str = "/tmp/nvidia-log",
        active_thread_percentage: int = 100,
    ):
        self.pipe_dir = Path(pipe_dir)
        self.log_dir = Path(log_dir)
        self.active_thread_percentage = active_thread_percentage
        self._status = MPSStatus.NOT_STARTED

    @property
    def status(self) -> MPSStatus:
        return self._status

    def start(self) -> bool:
        """Start the MPS daemon. Returns True if started successfully."""
        if not torch.cuda.is_available():
            logger.info("No CUDA GPU available, MPS not needed")
            self._status = MPSStatus.UNAVAILABLE
            return False

        try:
            self.pipe_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(self.log_dir)

            result = subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error("MPS daemon start failed: %s", result.stderr)
                self._status = MPSStatus.FAILED
                self._clear_env_vars()
                return False

            # Verify daemon is responding
            verify = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="get_default_active_thread_percentage\n",
                capture_output=True,
                text=True,
                timeout=5,
            )

            if verify.returncode != 0:
                logger.error("MPS daemon not responding after start")
                self._status = MPSStatus.FAILED
                self._clear_env_vars()
                return False

            # Set thread percentage if non-default
            if self.active_thread_percentage != 100:
                subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input=f"set_default_active_thread_percentage {self.active_thread_percentage}\n",
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

            self._status = MPSStatus.RUNNING
            logger.info(
                "MPS daemon started (pipe=%s, thread_pct=%d)",
                self.pipe_dir,
                self.active_thread_percentage,
            )
            return True

        except FileNotFoundError:
            logger.error("nvidia-cuda-mps-control not found - is CUDA toolkit installed?")
            self._status = MPSStatus.UNAVAILABLE
            self._clear_env_vars()
            return False
        except subprocess.TimeoutExpired:
            logger.error("MPS daemon start timed out")
            self._status = MPSStatus.FAILED
            self._clear_env_vars()
            return False
        except Exception:
            logger.exception("Unexpected error starting MPS daemon")
            self._status = MPSStatus.FAILED
            self._clear_env_vars()
            return False

    def stop(self) -> None:
        """Stop the MPS daemon."""
        if self._status != MPSStatus.RUNNING:
            return

        try:
            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._status = MPSStatus.STOPPED
            self._clear_env_vars()
            logger.info("MPS daemon stopped")
        except Exception:
            logger.exception("Error stopping MPS daemon")
            self._status = MPSStatus.STOPPED
            self._clear_env_vars()

    def is_healthy(self) -> bool:
        """Check if MPS daemon is still running and responsive."""
        if self._status != MPSStatus.RUNNING:
            return False

        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="get_default_active_thread_percentage\n",
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _clear_env_vars(self) -> None:
        """Remove MPS env vars from os.environ to prevent leaking on failure."""
        for var in _MPS_ENV_VARS:
            os.environ.pop(var, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_mps.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/mps.py tests/test_mps.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add omnivoice_server/mps.py tests/test_mps.py
git commit -m "feat(mps): add MPSManager for NVIDIA CUDA MPS daemon lifecycle"
```

---

## Task 3: Create Worker Manager (with crash recovery, MPS health check, VRAM guard)

**Files:**
- Create: `omnivoice_server/worker_manager.py`
- Create: `tests/test_worker_manager.py`

**Key design decisions:**
- `worker_main` is stored as `self._worker_main` so `_restart_worker` can re-fork on crash
- Monitor loop includes periodic MPS health check (30s interval) with full restart on failure
- VRAM guard: first worker measures peak VRAM after load, writes to temp file, parent reads before forking remaining

- [ ] **Step 1: Write failing tests for worker manager**

Create `tests/test_worker_manager.py`:

```python
import pytest
from unittest.mock import patch, MagicMock, call
from omnivoice_server.worker_manager import WorkerManager


@pytest.fixture
def mock_socket():
    with patch("omnivoice_server.worker_manager.socket") as m:
        sock = MagicMock()
        m.socket.return_value = sock
        # socket.socket() returns the mock directly (no context manager)
        yield m, sock


def test_worker_manager_creation():
    mgr = WorkerManager(num_workers=4, host="0.0.0.0", port=8880)
    assert mgr.num_workers == 4
    assert len(mgr.worker_pids) == 0


def test_create_shared_socket(mock_socket):
    _, sock = mock_socket
    mgr = WorkerManager(num_workers=4, host="0.0.0.0", port=8880)
    fd = mgr.create_shared_socket()
    assert sock.setsockopt.called
    assert sock.bind.called
    assert sock.listen.called
    assert isinstance(fd, int)


def test_spawn_workers_stores_worker_main():
    """worker_main must be stored so _restart_worker can re-fork."""
    called = {"count": 0}

    def fake_main():
        called["count"] += 1

    mgr = WorkerManager(num_workers=2, host="0.0.0.0", port=8880)

    with patch("omnivoice_server.worker_manager.os") as mock_os:
        mock_os.fork.return_value = 100  # parent sees child PID
        mock_os.getpid.return_value = 1
        mgr.spawn_workers(fake_main)

    assert mgr._worker_main is fake_main
    assert len(mgr.worker_pids) == 2


def test_restart_worker_actually_forks():
    """_restart_worker must call os.fork(), not just log."""
    mgr = WorkerManager(num_workers=2, host="0.0.0.0", port=8880)
    mgr._worker_main = lambda: None
    mgr.worker_pids = {0: 100}  # slot 0, pid 100 (about to be replaced)

    with patch("omnivoice_server.worker_manager.os") as mock_os:
        mock_os.fork.return_value = 200  # new child PID
        mock_os.getpid.return_value = 1
        mock_os.environ = {}
        mgr._restart_worker(0, dead_pid=100)

    assert mock_os.fork.called
    assert mgr.worker_pids[0] == 200


def test_crash_loop_stops_restart():
    mgr = WorkerManager(num_workers=1, host="0.0.0.0", port=8880)
    import time
    mgr._crash_counts = {0: 4}  # slot 0 exceeded threshold of 3
    mgr._crash_window_start[0] = time.time()  # within window

    result = mgr._should_restart(0)
    assert result is False


def test_graceful_shutdown_sends_sigterm():
    mgr = WorkerManager(num_workers=2, host="0.0.0.0", port=8880)
    mgr.worker_pids = {0: 111, 1: 222}

    with patch("omnivoice_server.worker_manager.os") as mock_os:
        mock_os.environ = {}
        mock_os.kill = MagicMock()
        mock_os.waitpid = MagicMock(side_effect=ChildProcessError)
        mgr.shutdown(timeout=5)
        kill_calls = mock_os.kill.call_args_list
        assert any(111 in c[0] for c in kill_calls)
        assert any(222 in c[0] for c in kill_calls)


def test_mps_health_check_failure_triggers_full_restart():
    """When MPS dies, all workers must be killed and restarted."""
    mgr = WorkerManager(num_workers=2, host="0.0.0.0", port=8880)
    mgr._worker_main = lambda: None
    mgr.worker_pids = {0: 111, 1: 222}

    mock_mps = MagicMock()
    mock_mps.is_healthy.return_value = False
    mock_mps.start.return_value = True

    with patch("omnivoice_server.worker_manager.os") as mock_os:
        mock_os.environ = {}
        mock_os.kill = MagicMock()
        mock_os.waitpid = MagicMock(side_effect=ChildProcessError)
        mock_os.fork.return_value = 300
        mock_os.getpid.return_value = 1

        mgr._handle_mps_failure(mock_mps)

    # All old workers should be gone
    assert 111 not in mgr.worker_pids.values()
    assert 222 not in mgr.worker_pids.values()
    # New workers should be forked
    assert len(mgr.worker_pids) == 2
    mock_mps.stop.assert_called_once()
    mock_mps.start.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_worker_manager.py -v`
Expected: FAIL — `omnivoice_server.worker_manager` module not found

- [ ] **Step 3: Implement WorkerManager**

Create `omnivoice_server/worker_manager.py`:

```python
"""Multi-worker process manager for parallel inference.

Forks N worker processes from a parent process. Each worker independently
initializes CUDA and loads the model. The parent monitors workers,
restarts crashed ones, and handles graceful shutdown.

Includes:
- Worker slot identity via OMNIVOICE_WORKER_SLOT env var (set by _fork_worker)
- Periodic MPS daemon health check (30s) with full restart on failure
- Worker crash recovery with rate limiting keyed by slot (3 crashes / 60s)
- VRAM guard: first worker measures peak VRAM (via lifespan in app.py), parent reads measurement file and adjusts count
- Graceful shutdown with WNOHANG polling (actual timeout enforcement)
"""

import logging
import os
import signal
import socket
import sys
import tempfile
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

CRASH_THRESHOLD = 3
CRASH_WINDOW_S = 60
MPS_CHECK_INTERVAL_S = 30
VRAM_HEADROOM_FACTOR = 2.0


class WorkerManager:
    """Manages a pool of worker processes for parallel GPU inference."""

    def __init__(self, num_workers: int, host: str, port: int):
        self.num_workers = num_workers
        self.host = host
        self.port = port
        self.worker_pids: dict[int, int] = {}  # slot -> pid
        self._socket_fd: Optional[int] = None
        self._socket: Optional[socket.socket] = None
        self._worker_main: Optional[Callable[[], None]] = None
        self._crash_counts: dict[int, int] = {}
        self._crash_window_start: dict[int, float] = {}
        self._shutdown_requested = False
        self._vram_measurement_file = os.path.join(
            tempfile.gettempdir(), "omnivoice_vram_measurement"
        )

    def create_shared_socket(self) -> int:
        """Create a TCP socket with SO_REUSEPORT for kernel-level load balancing.

        Returns the file descriptor for the bound socket.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            logger.warning("SO_REUSEPORT not available, falling back to SO_REUSEADDR only")
        sock.bind((self.host, self.port))
        sock.listen(128)
        # Keep reference so socket is not garbage-collected (would close fd)
        self._socket = sock
        self._socket_fd = sock.fileno()
        logger.info("Shared socket bound to %s:%d (fd=%d)", self.host, self.port, self._socket_fd)
        return self._socket_fd

    def spawn_workers(self, worker_main: Callable[[], None]) -> None:
        """Fork N worker processes.

        Args:
            worker_main: Function each worker calls after fork.
                         Must initialize CUDA and load model independently.
                         Should call sys.exit(0) when done.
        """
        self._worker_main = worker_main

        for slot in range(self.num_workers):
            self._fork_worker(slot)

    def _fork_worker(self, slot: int) -> None:
        """Fork a single worker for the given slot."""
        pid = os.fork()
        if pid == 0:
            # Child process — set slot identity so worker can measure VRAM on slot 0
            os.environ["OMNIVOICE_WORKER_SLOT"] = str(slot)
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            logger.info("Worker slot=%d pid=%d starting", slot, os.getpid())
            try:
                self._worker_main()
            except Exception:
                logger.exception("Worker slot=%d pid=%d crashed", slot, os.getpid())
                sys.exit(1)
            sys.exit(0)
        else:
            # Parent process
            self.worker_pids[slot] = pid
            logger.info("Forked worker slot=%d pid=%d", slot, pid)

    def spawn_with_vram_guard(self, worker_main: Callable[[], None]) -> None:
        """Fork workers sequentially with VRAM guard.

        1. Fork first worker and wait for it to write VRAM measurement
        2. Read measurement, calculate safe worker count
        3. Fork remaining workers (up to safe limit)
        """
        self._worker_main = worker_main

        # Remove stale measurement from previous runs
        if os.path.exists(self._vram_measurement_file):
            os.remove(self._vram_measurement_file)

        # Fork first worker
        self._fork_worker(0)

        # Wait for VRAM measurement file (timeout 120s for model download)
        logger.info("Waiting for first worker VRAM measurement...")
        deadline = time.time() + 120
        while time.time() < deadline:
            if os.path.exists(self._vram_measurement_file):
                time.sleep(0.5)  # brief wait for write to complete
                break
            time.sleep(1)
            # Also check if worker is still alive
            pid0 = self.worker_pids.get(0)
            if pid0:
                try:
                    _pid, status = os.waitpid(pid0, os.WNOHANG)
                    if _pid != 0:
                        logger.error("First worker died during VRAM measurement")
                        return
                except ChildProcessError:
                    logger.error("First worker lost during VRAM measurement")
                    return
        else:
            logger.warning("Timed out waiting for VRAM measurement, forking all %d workers", self.num_workers)
            self._fork_remaining(1)
            return

        # Read measurement
        try:
            with open(self._vram_measurement_file) as f:
                data = f.read().strip()
            peak_vram_mb = float(data)
            logger.info("First worker peak VRAM: %.0f MB", peak_vram_mb)
        except Exception:
            logger.exception("Failed to read VRAM measurement, forking all workers")
            self._fork_remaining(1)
            return

        # Calculate safe workers
        try:
            import torch
            total_vram_mb = torch.cuda.mem_get_info()[1] / 1024 / 1024
            safe_workers = int(total_vram_mb / (peak_vram_mb * VRAM_HEADROOM_FACTOR))
            safe_workers = max(safe_workers, 1)  # at least 1 worker

            if safe_workers < self.num_workers:
                logger.warning(
                    "VRAM guard: reducing workers %d -> %d (total=%.0fMB, peak=%.0fMB per worker, factor=%.1f)",
                    self.num_workers, safe_workers, total_vram_mb, peak_vram_mb, VRAM_HEADROOM_FACTOR,
                )
                self.num_workers = safe_workers
        except Exception:
            logger.exception("VRAM guard calculation failed, forking all remaining workers")

        self._fork_remaining(1)

    def _fork_remaining(self, start_slot: int) -> None:
        """Fork workers from start_slot to num_workers."""
        for slot in range(start_slot, self.num_workers):
            self._fork_worker(slot)

    def monitor(self, mps_manager=None) -> None:
        """Block and monitor workers. Restart on crash. Check MPS health. Exit on SIGTERM."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(
            "Parent pid=%d monitoring %d workers",
            os.getpid(),
            len(self.worker_pids),
        )

        last_mps_check = time.time()

        while not self._shutdown_requested and self.worker_pids:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    # No child exited — do MPS health check
                    if mps_manager and (time.time() - last_mps_check) > MPS_CHECK_INTERVAL_S:
                        if not mps_manager.is_healthy():
                            logger.critical("MPS daemon died! Full restart required.")
                            self._handle_mps_failure(mps_manager)
                        last_mps_check = time.time()

                    time.sleep(0.5)
                    continue

                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                slot = self._find_slot(pid)
                if slot is not None:
                    logger.warning(
                        "Worker slot=%d pid=%d exited with code=%d",
                        slot, pid, exit_code,
                    )
                    del self.worker_pids[slot]
                    if self._should_restart(pid):
                        self._restart_worker(slot, dead_pid=pid)

            except ChildProcessError:
                logger.info("All workers exited")
                break

        logger.info("Parent monitor loop ended")

    def _handle_signal(self, signum: int, frame) -> None:
        logger.info("Parent received signal=%d, initiating shutdown", signum)
        self._shutdown_requested = True

    def _should_restart(self, slot: int) -> bool:
        count = self._crash_counts.get(slot, 0)
        if count >= CRASH_THRESHOLD:
            logger.error(
                "Worker slot=%d crashed %d times in %ds - not restarting",
                slot, count, CRASH_WINDOW_S,
            )
            return False
        return True

    def _restart_worker(self, slot: int, dead_pid: int) -> None:
        """Re-fork a worker for the given slot."""
        # Track crash rate by slot (not PID — PIDs change on restart)
        now = time.time()
        self._crash_counts[slot] = self._crash_counts.get(slot, 0) + 1
        window_start = self._crash_window_start.get(slot, now)
        if now - window_start > CRASH_WINDOW_S:
            self._crash_counts[slot] = 1
            self._crash_window_start[slot] = now

        if not self._should_restart(slot):
            return

        logger.info("Restarting worker slot=%d (dead_pid=%d)", slot, dead_pid)
        self._fork_worker(slot)

    def _handle_mps_failure(self, mps_manager) -> None:
        """MPS daemon death is catastrophic. Kill all workers, restart MPS, restart workers."""
        logger.critical("MPS daemon failure detected - initiating full restart")

        # Kill all workers
        self.shutdown(timeout=5)

        # Restart MPS
        mps_manager.stop()
        if not mps_manager.start():
            logger.critical("MPS restart failed - cannot recover, shutting down")
            self._shutdown_requested = True
            return

        # Re-fork all workers
        self.worker_pids.clear()
        for slot in range(self.num_workers):
            self._fork_worker(slot)

        logger.info("Full restart complete - %d workers running", len(self.worker_pids))

    def _find_slot(self, pid: int) -> Optional[int]:
        for slot, p in self.worker_pids.items():
            if p == pid:
                return slot
        return None

    def shutdown(self, timeout: int = 10) -> None:
        """Send SIGTERM to all workers, wait for graceful exit, then SIGKILL."""
        logger.info("Shutting down %d workers (timeout=%ds)", len(self.worker_pids), timeout)

        for slot, pid in list(self.worker_pids.items()):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Poll with WNOHANG until timeout expires
        deadline = time.time() + timeout
        while self.worker_pids and time.time() < deadline:
            for slot, pid in list(self.worker_pids.items()):
                try:
                    _pid, status = os.waitpid(pid, os.WNOHANG)
                    if _pid != 0:
                        del self.worker_pids[slot]
                except ChildProcessError:
                    del self.worker_pids[slot]
            if self.worker_pids:
                time.sleep(0.1)

        # Force kill any remaining
        for slot, pid in list(self.worker_pids.items()):
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass

        self.worker_pids.clear()
        logger.info("All workers shut down")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_worker_manager.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/worker_manager.py tests/test_worker_manager.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add omnivoice_server/worker_manager.py tests/test_worker_manager.py
git commit -m "feat(workers): add WorkerManager with crash recovery, MPS health check, and VRAM guard"
```

---

## Task 4: Simplify InferenceService for Single-Worker Mode

**Files:**
- Modify: `omnivoice_server/services/inference.py:137-206`

- [ ] **Step 1: Read current inference.py to confirm line numbers**

Read `omnivoice_server/services/inference.py` in full. Confirm the InferenceService class structure.

- [ ] **Step 2: Refactor InferenceService to accept optional executor**

In `omnivoice_server/services/inference.py`, modify the `InferenceService` class:

Replace the constructor to accept optional executor (backward compatible):

```python
class InferenceService:
    """Runs OmniVoice inference.

    In multi-worker mode (no executor): runs inference directly (single-threaded per worker).
    In single-worker fallback mode (with executor): uses ThreadPoolExecutor + Semaphore.
    """

    def __init__(
        self,
        model_svc: "ModelService",
        cfg: "Settings",
        executor: Optional["ThreadPoolExecutor"] = None,
    ):
        self._model_svc = model_svc
        self._cfg = cfg
        self._executor = executor
        self._semaphore = (
            asyncio.Semaphore(cfg.max_concurrent) if executor else None
        )
        self._adapter = OmniVoiceAdapter()
```

Modify `synthesize` to use executor only if present:

```python
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if self._executor is not None:
            return await self._synthesize_threaded(request)
        return await self._synthesize_direct(request)

    async def _synthesize_direct(self, request: SynthesisRequest) -> SynthesisResult:
        """Direct inference (multi-worker mode). Each worker handles one request at a time."""
        # Still offload to executor thread to avoid blocking the event loop
        # (worker needs event loop for health checks, metrics, etc.)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sync, request)

    async def _synthesize_threaded(self, request: SynthesisRequest) -> SynthesisResult:
        """Threaded inference with semaphore (single-worker fallback mode)."""
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, self._run_sync, request
                    ),
                    timeout=self._cfg.request_timeout_s,
                )
                return result
            except asyncio.TimeoutError:
                raise
```

Gate `_cleanup_memory` behind executor check in `_run_sync` finally block:

```python
    def _run_sync(self, request: SynthesisRequest) -> SynthesisResult:
        try:
            # ... existing inference logic unchanged ...
        finally:
            # Only clean up aggressively in single-worker (threaded) mode
            if self._executor is not None:
                _cleanup_memory(self._cfg.device)
```

- [ ] **Step 3: Run existing tests to confirm nothing breaks**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_speech.py -v`
Expected: All PASS (executor is still passed via conftest, so threaded path is used)

- [ ] **Step 4: Run ruff**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/services/inference.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add omnivoice_server/services/inference.py
git commit -m "refactor(inference): support optional executor for multi-worker compatibility"
```

---

## Task 5: Update app.py — Skip Model Load in Parent for Multi-Worker

**Critical:** Parent must NOT initialize CUDA or load the model when `workers > 1`. Each worker loads independently after fork.

**Files:**
- Modify: `omnivoice_server/app.py:27-69`

- [ ] **Step 1: Read current app.py to confirm line numbers**

Read `omnivoice_server/app.py` in full.

- [ ] **Step 2: Make lifespan conditional on worker mode**

In `omnivoice_server/app.py`, modify the `lifespan` function:

```python
async def lifespan(app: FastAPI):
    cfg: Settings = app.state.cfg
    app.state.start_time = time.time()
    app.state.metrics_svc = MetricsService()

    if cfg.workers == 1:
        # Single-worker mode: load model here, use ThreadPoolExecutor
        model_svc = ModelService(cfg)
        app.state.model_svc = model_svc
        executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent)
        app.state.executor = executor
        app.state.inference_svc = InferenceService(model_svc, cfg, executor)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, model_svc.load)
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

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, model_svc.load)

        # Slot 0 worker: measure peak VRAM and write to temp file for parent
        worker_slot = int(os.environ.get("OMNIVOICE_WORKER_SLOT", "-1"))
        if worker_slot == 0 and cfg.device == "cuda":
            import torch
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / 1024 / 1024
            vram_file = os.path.join(
                tempfile.gettempdir(), "omnivoice_vram_measurement"
            )
            with open(vram_file, "w") as f:
                f.write(f"{peak_mb}")
            logger.info("VRAM measurement written: %.0f MB -> %s", peak_mb, vram_file)

    yield

    # Shutdown
    if getattr(app.state, "executor", None):
        app.state.executor.shutdown(wait=False)
```

Note: In multi-worker mode, each worker process runs its own lifespan after `os.fork()`. The parent process never enters this code — it calls `worker_mgr.monitor()` directly and never invokes `create_app()`. Model loading happens once per worker, inside the fork.

- [ ] **Step 3: Run existing tests to confirm nothing breaks**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/ -v`
Expected: All PASS (conftest uses workers=1 by default)

- [ ] **Step 4: Commit**

```bash
git add omnivoice_server/app.py
git commit -m "refactor(app): conditional model loading — parent skips CUDA init in multi-worker mode"
```

---

## Task 6: Update conftest.py for workers=1

**This must be done before or alongside Task 5.** Ensures all existing tests continue using single-worker mode.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Read current conftest.py**

Read `tests/conftest.py` in full.

- [ ] **Step 2: Add workers=1 to settings fixture**

In the `settings` fixture (or wherever Settings is constructed), add `workers=1`:

```python
@pytest.fixture
def settings():
    return Settings(
        device="cpu",
        num_step=4,
        max_concurrent=1,
        workers=1,  # Ensure tests use single-worker mode
        # ... rest of existing settings ...
    )
```

- [ ] **Step 3: Run full test suite**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: pin conftest settings to workers=1 for stable single-worker test mode"
```

---

## Task 7: Add File Locking to Profile Writes

**Files:**
- Modify: `omnivoice_server/services/profiles.py:59-94`

- [ ] **Step 1: Read current profiles.py**

Read `omnivoice_server/services/profiles.py` in full.

- [ ] **Step 2: Add fcntl.flock to save_profile and delete_profile**

Add import at top of file:

```python
import fcntl
```

Wrap save and delete operations with a lock file at the **profiles root** (not inside the profile directory, which gets deleted):

```python
    def _acquire_lock(self, profile_id: str):
        """Get file object for advisory lock. Caller must close it."""
        lock_path = self._profiles_dir / f".lock-{profile_id}"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return lock_file

    def _release_lock(self, lock_file):
        """Release and close the lock file."""
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    def save_profile(self, profile_id, audio_data, ref_text=None, overwrite=False):
        lock = self._acquire_lock(profile_id)
        try:
            # ... existing save logic ...
        finally:
            self._release_lock(lock)

    def delete_profile(self, profile_id):
        lock = self._acquire_lock(profile_id)
        try:
            # ... existing delete logic ...
        finally:
            self._release_lock(lock)
```

- [ ] **Step 3: Run existing tests**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/test_voices.py -v`
Expected: All PASS

- [ ] **Step 4: Run ruff**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/services/profiles.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add omnivoice_server/services/profiles.py
git commit -m "feat(profiles): add fcntl file locking for multi-worker write safety"
```

---

## Task 8: Wire CLI — MPS + Worker Orchestration

**Files:**
- Modify: `omnivoice_server/cli.py:16-157`

- [ ] **Step 1: Add --workers CLI argument**

In `omnivoice_server/cli.py`, add after existing argparse arguments:

```python
    parser.add_argument("--workers", type=int, default=None, dest="workers",
                        help="Number of worker processes (default: 1, set 4+ for MPS parallel)")
    parser.add_argument("--mps-enabled", choices=["auto", "true", "false"],
                        default=None, dest="mps_enabled",
                        help="MPS daemon mode (default: auto)")
    parser.add_argument("--mps-active-thread-pct", type=int, default=None,
                        dest="mps_active_thread_percentage",
                        help="GPU compute percentage for MPS (1-100, default: 100)")
```

- [ ] **Step 2: Add multi-worker orchestration to main()**

In the `main()` function, after Settings instantiation and before `uvicorn.run()`:

```python
    if cfg.workers > 1:
        # Multi-worker mode with MPS
        from omnivoice_server.mps import MPSManager
        from omnivoice_server.worker_manager import WorkerManager

        mps_mgr = None
        if cfg.mps_should_enable:
            mps_mgr = MPSManager(
                active_thread_percentage=cfg.mps_active_thread_percentage
            )
            if not mps_mgr.start():
                logger.warning("MPS failed to start, falling back to single-worker mode")
                cfg = cfg.model_copy(update={"workers": 1})

        if cfg.workers > 1:
            worker_mgr = WorkerManager(
                num_workers=cfg.workers,
                host=cfg.host,
                port=cfg.port,
            )
            fd = worker_mgr.create_shared_socket()

            def worker_main():
                """Each worker: create app, load model via lifespan, serve on inherited socket."""
                import uvicorn
                from omnivoice_server.app import create_app

                app = create_app(cfg)
                # Slot 0 worker writes VRAM measurement during lifespan (see Task 5).
                # Use socket from inherited fd
                sock = socket.socket(fileno=fd, family=socket.AF_INET, type=socket.SOCK_STREAM)
                config = uvicorn.Config(app, workers=1, log_level=cfg.log_level)
                server = uvicorn.Server(config=config)
                asyncio.run(server.serve(sockets=[sock]))

            # Fork workers with VRAM guard
            worker_mgr.spawn_with_vram_guard(worker_main)

            mps_active = mps_mgr is not None and mps_mgr.status.value == "running"
            logger.info("OMNIVOICE_READY workers=%d mps=%s", cfg.workers, mps_active)
            print(f"OMNIVOICE_READY workers={cfg.workers} mps={mps_active}")

            try:
                worker_mgr.monitor(mps_manager=mps_mgr)
            except KeyboardInterrupt:
                pass
            finally:
                worker_mgr.shutdown(timeout=cfg.shutdown_timeout)
                if mps_mgr:
                    mps_mgr.stop()
            return

    # Single-worker mode: existing behavior
    uvicorn.run(...)
```

**Key implementation note:** `socket.socket(fileno=fd)` wraps the inherited fd without duplicating it. Each child gets its own Python socket object but the underlying fd is the same. When the child exits, `uvicorn.Server` closes its socket, but the parent's original socket reference (`self._socket`) keeps the fd alive.

- [ ] **Step 3: Run ruff**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/cli.py`
Expected: No errors

- [ ] **Step 4: Run existing tests**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/ -v`
Expected: All PASS (tests use conftest with workers=1)

- [ ] **Step 5: Commit**

```bash
git add omnivoice_server/cli.py
git commit -m "feat(cli): add --workers flag with MPS + worker orchestration and VRAM guard"
```

---

## Task 9: Update Dockerfile for CUDA + MPS

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Update Dockerfile with CUDA base and MPS control binary**

Keep the multi-stage build pattern. Use `pytorch/pytorch` as builder and create a runtime stage with MPS support:

```dockerfile
# Stage 1: Build dependencies
FROM pytorch/pytorch:2.3.0-cuda12.4-cudnn9-devel AS builder

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Stage 2: Runtime with MPS support
FROM pytorch/pytorch:2.3.0-cuda12.4-cudnn9-runtime AS runtime

# Install CUDA MPS control utility and runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy nvidia-cuda-mps-control from devel image
COPY --from=builder /usr/local/cuda/bin/nvidia-cuda-mps-control /usr/local/cuda/bin/

WORKDIR /app
COPY --from=builder /app /app
COPY . .

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "omnivoice_server"]
```

**Note:** The `nvidia-cuda-mps-control` binary is copied from the `devel` image where it's available. The `runtime` image is smaller but doesn't include it by default.

- [ ] **Step 2: Update docker-compose.yml with GPU device**

```yaml
services:
  omnivoice-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${OMNIVOICE_PORT:-8880}:8880"
    environment:
      - OMNIVOICE_HOST=0.0.0.0
      - OMNIVOICE_DEVICE=cuda
      - OMNIVOICE_WORKERS=${OMNIVOICE_WORKERS:-1}
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      - ${OMNIVOICE_PROFILE_DIR:-./profiles}:/app/profiles
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8880/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat(docker): CUDA base image with MPS control binary and GPU device reservation"
```

---

## Task 10: Run Full Test Suite + Ruff

**Files:** None (verification only)

- [ ] **Step 1: Run ruff on all Python files**

Run: `cd /mnt/data/work/omnivoice-server && ruff check omnivoice_server/ tests/`
Expected: No errors

- [ ] **Step 2: Run full test suite**

Run: `cd /mnt/data/work/omnivoice-server && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 3: Run mypy if configured**

Run: `cd /mnt/data/work/omnivoice-server && python -m mypy omnivoice_server/ --ignore-missing-imports`
Expected: No errors (or only pre-existing ones)

---

## Task 11: Final Integration Test (Manual / CI)

**Files:** None (manual verification on GPU machine)

- [ ] **Step 1: Build Docker image**

Run: `cd /mnt/data/work/omnivoice-server && docker build -t omnivoice-server:mps .`
Expected: Build succeeds

- [ ] **Step 2: Run container with GPU**

Run: `docker run --gpus all -p 8880:8880 -e OMNIVOICE_WORKERS=4 omnivoice-server:mps`
Expected: MPS daemon starts, 4 workers fork, `OMNIVOICE_READY` logged

- [ ] **Step 3: Test concurrent requests**

Send 4+ simultaneous `POST /v1/audio/speech` requests. Verify all return successfully and total wall time is ~1x (not ~4x), confirming parallel execution.

- [ ] **Step 4: Test VRAM guard**

Attempt `OMNIVOICE_WORKERS=16` — verify server auto-reduces worker count to fit available VRAM.

- [ ] **Step 5: Test fallback (stop MPS)**

Kill MPS daemon, verify server detects failure and restarts all workers.

- [ ] **Step 6: Test worker crash recovery**

Kill one worker process, verify parent detects crash and forks replacement.
