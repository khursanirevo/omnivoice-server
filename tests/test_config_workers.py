import pytest

from omnivoice_server.config import Settings


def test_workers_default_is_one():
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
