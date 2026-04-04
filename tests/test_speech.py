"""
Tests for speech synthesis endpoints.
"""
from __future__ import annotations


def test_speech_auto_returns_wav(client):
    """Auto voice mode returns WAV with RIFF header."""
    resp = client.post("/v1/audio/speech", json={
        "model": "omnivoice",
        "input": "Hello world",
        "voice": "auto",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    # WAV magic bytes: RIFF
    assert resp.content[:4] == b"RIFF"


def test_speech_design_voice(client):
    """Design voice with attributes."""
    resp = client.post("/v1/audio/speech", json={
        "input": "Hello",
        "voice": "design:female,british accent",
    })
    assert resp.status_code == 200


def test_speech_invalid_text_empty(client):
    """Empty text returns 422."""
    resp = client.post("/v1/audio/speech", json={
        "input": "",
        "voice": "auto",
    })
    assert resp.status_code == 422


def test_speech_clone_unknown_profile(client):
    """Unknown profile returns 404."""
    resp = client.post("/v1/audio/speech", json={
        "input": "Hello",
        "voice": "clone:nonexistent",
    })
    assert resp.status_code == 404


def test_speech_openai_model_names_accepted(client):
    """tts-1 and tts-1-hd should be accepted for drop-in compatibility."""
    for model_name in ("tts-1", "tts-1-hd", "omnivoice"):
        resp = client.post("/v1/audio/speech", json={
            "model": model_name,
            "input": "Hello",
        })
        assert resp.status_code == 200, f"Failed for model={model_name}"


def test_speech_pcm_format(client):
    """response_format=pcm returns audio/pcm."""
    resp = client.post("/v1/audio/speech", json={
        "input": "Hello",
        "response_format": "pcm",
    })
    assert resp.status_code == 200
    assert "audio/pcm" in resp.headers["content-type"]
