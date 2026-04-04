"""
Tests for one-shot voice cloning endpoint.
"""
from __future__ import annotations

import io


def test_clone_returns_wav(client, sample_audio_bytes):
    """POST /v1/audio/speech/clone returns WAV."""
    resp = client.post(
        "/v1/audio/speech/clone",
        data={"text": "Hello world", "speed": "1.0"},
        files={"ref_audio": ("ref.wav", io.BytesIO(sample_audio_bytes), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


def test_clone_empty_audio_rejected(client):
    """Empty audio returns 422."""
    resp = client.post(
        "/v1/audio/speech/clone",
        data={"text": "Hello"},
        files={"ref_audio": ("ref.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert resp.status_code == 422
