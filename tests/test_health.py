"""
Tests for health and metrics endpoints.
"""
from __future__ import annotations


def test_health_ok(client):
    """GET /health returns 200 with status, device, uptime_s."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "uptime_s" in data


def test_metrics_ok(client):
    """GET /metrics returns 200 with requests_total, ram_mb."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "requests_total" in data
    assert "ram_mb" in data
