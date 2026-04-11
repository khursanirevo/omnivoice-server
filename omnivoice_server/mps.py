"""NVIDIA CUDA MPS (Multi-Process Service) daemon lifecycle management."""

import logging
import os
import subprocess
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_MPS_ENV_VARS = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")


def _cuda_gpu_available() -> bool:
    """Check if a CUDA GPU exists WITHOUT initializing the CUDA driver.

    torch.cuda.is_available() initializes CUDA in the calling process,
    which breaks fork() — child processes get "Cannot re-initialize CUDA
    in forked subprocess". Use nvidia-smi instead, which is a separate
    process and leaves the parent's CUDA state untouched.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class MPSStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class MPSManager:
    """Manages the NVIDIA CUDA MPS daemon lifecycle."""

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
        if not _cuda_gpu_available():
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
            # Daemon may already be running — that's fine
            already_running = (
                "already running" in (result.stderr or "").lower()
                or "already running" in (result.stdout or "").lower()
            )
            if result.returncode != 0 and not already_running:
                self._status = MPSStatus.FAILED
                self._clear_env_vars()
                return False

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
            logger.error(
                "nvidia-cuda-mps-control not found - is CUDA toolkit installed?"
            )
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
        for var in _MPS_ENV_VARS:
            os.environ.pop(var, None)
