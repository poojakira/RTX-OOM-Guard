from fastapi.testclient import TestClient
from rtx_oom_guard.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "rtx_oom_guard-api"}

def test_get_telemetry():
    response = client.get("/api/memory")
    assert response.status_code == 200
    data = response.json()
    assert "current_allocated_mb" in data
    assert "current_reserved_mb" in data
    assert "free_estimate_mb" in data
    assert "current_frag" in data

