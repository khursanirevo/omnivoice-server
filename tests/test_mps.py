from unittest.mock import MagicMock, patch

import pytest

from omnivoice_server.mps import MPSManager, MPSStatus


@pytest.fixture
def mock_subprocess():
    with patch("omnivoice_server.mps.subprocess") as mock_sub:
        # Default: nvidia-smi succeeds, mps-control succeeds
        mock_sub.run.return_value = MagicMock(returncode=0, stdout="NVIDIA H200\n")
        yield mock_sub


@pytest.fixture
def mock_gpu_available():
    with patch("omnivoice_server.mps._cuda_gpu_available", return_value=True):
        yield


@pytest.fixture
def mock_gpu_unavailable():
    with patch("omnivoice_server.mps._cuda_gpu_available", return_value=False):
        yield


def test_status_not_started():
    mgr = MPSManager()
    assert mgr.status == MPSStatus.NOT_STARTED


def test_start_success(mock_subprocess, mock_gpu_available, tmp_path):
    mgr = MPSManager(pipe_dir=str(tmp_path / "mps"), log_dir=str(tmp_path / "log"))
    result = mgr.start()
    assert result is True
    assert mgr.status == MPSStatus.RUNNING


def test_start_no_gpu(mock_subprocess, mock_gpu_unavailable):
    mgr = MPSManager()
    result = mgr.start()
    assert result is False
    assert mgr.status == MPSStatus.UNAVAILABLE


def test_start_mps_control_fails(mock_subprocess, mock_gpu_available, tmp_path):
    mock_subprocess.run.return_value = MagicMock(returncode=1, stdout="")
    mgr = MPSManager(pipe_dir=str(tmp_path / "mps"), log_dir=str(tmp_path / "log"))
    result = mgr.start()
    assert result is False
    assert mgr.status == MPSStatus.FAILED


def test_stop(mock_subprocess, mock_gpu_available, tmp_path):
    mgr = MPSManager(pipe_dir=str(tmp_path / "mps"), log_dir=str(tmp_path / "log"))
    mgr.start()
    mgr.stop()
    assert mgr.status == MPSStatus.STOPPED


def test_health_check_running(mock_subprocess, mock_gpu_available, tmp_path):
    mgr = MPSManager(pipe_dir=str(tmp_path / "mps"), log_dir=str(tmp_path / "log"))
    mgr.start()
    assert mgr.is_healthy() is True


def test_health_check_daemon_died(mock_subprocess, mock_gpu_available, tmp_path):
    mgr = MPSManager(pipe_dir=str(tmp_path / "mps"), log_dir=str(tmp_path / "log"))
    mgr.start()
    mock_subprocess.run.side_effect = Exception("daemon not responding")
    assert mgr.is_healthy() is False


def test_set_thread_percentage(mock_subprocess, mock_gpu_available, tmp_path):
    mgr = MPSManager(
        pipe_dir=str(tmp_path / "mps"),
        log_dir=str(tmp_path / "log"),
        active_thread_percentage=75,
    )
    mgr.start()
    calls = mock_subprocess.run.call_args_list
    set_call = [c for c in calls if "set_default_active_thread_percentage" in str(c)]
    assert len(set_call) >= 1


def test_env_vars_cleared_on_failure(mock_subprocess, mock_gpu_available, tmp_path):
    import os

    mock_subprocess.run.return_value = MagicMock(returncode=1, stdout="")
    mgr = MPSManager(pipe_dir=str(tmp_path / "mps"), log_dir=str(tmp_path / "log"))
    mgr.start()
    assert mgr.status == MPSStatus.FAILED
    assert os.environ.get("CUDA_MPS_PIPE_DIRECTORY") is None
