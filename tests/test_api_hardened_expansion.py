from fastapi.testclient import TestClient
from unittest.mock import patch
from rtx_oom_guard.api import app

def test_api_telemetry_retry_failure():
    """Verify API handles persistent telemetry read failures."""
    client = TestClient(app)
    # Patch Path to exist but open to fail
    with patch("rtx_oom_guard.api.Path.exists", return_value=True), \
         patch("builtins.open", side_effect=IOError("Locked")):
        response = client.get("/api/telemetry")
        assert response.status_code == 200
        assert response.json()["total_compactions"] == 0

def test_api_benchmarks_io_error():
    """Verify API handles IO errors in benchmark retrieval (Line 163)."""
    client = TestClient(app)
    with patch("rtx_oom_guard.api.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", side_effect=ValueError("Disk error")):
        response = client.get("/api/benchmarks")
        assert response.status_code == 500

def test_api_risk_scoring_tiers():
    """Verify risk scoring tiers (SAFE, WARN, ACT)."""
    client = TestClient(app)
    # Patch the score method of the risk model instance
    with patch("rtx_oom_guard.api._risk_model.score") as mock_score:
        # tier SAFE
        mock_score.return_value = 0.1
        response = client.post("/api/risk", json={"fragmentation": 0.1, "utilisation": 0.5, "alloc_delta_mb": 0.0})
        assert response.json()["tier"] == "SAFE"
    
        # tier WARN
        mock_score.return_value = 0.6
        response = client.post("/api/risk", json={"fragmentation": 0.6, "utilisation": 0.5, "alloc_delta_mb": 0.0})
        assert response.json()["tier"] == "WARN"
        
        # tier ACT
        mock_score.return_value = 0.9
        response = client.post("/api/risk", json={"fragmentation": 0.9, "utilisation": 0.5, "alloc_delta_mb": 0.0})
        assert response.json()["tier"] == "ACT"

def test_gpu_snapshot_success():
    """Verify _gpu_snapshot when CUDA is available (Line 65-67)."""
    from rtx_oom_guard.api import _gpu_snapshot
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 100), \
         patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 200):
        res = _gpu_snapshot()
        assert res["cuda_available"] is True
        assert res["current_allocated_mb"] == 100.0
        assert res["current_reserved_mb"] == 200.0
        assert res["current_frag"] == 0.5

def test_gpu_snapshot_import_error():
    """Verify _gpu_snapshot catch-all for imports (Line 74-75)."""
    from rtx_oom_guard.api import _gpu_snapshot
    with patch("torch.cuda.is_available", side_effect=ImportError("No torch")):
        res = _gpu_snapshot()
        assert res["cuda_available"] is False

def test_api_risk_history_cleared():
    """Verify risk history returns empty list after clear_history (Line 117-120)."""
    from rtx_oom_guard.api import _risk_model
    _risk_model.clear_history()
    client = TestClient(app)
    response = client.get("/api/risk/history")
    assert response.status_code == 200
    assert response.json()["count"] == 0

def test_api_benchmarks_no_dir():
    """Verify benchmarks endpoint handled missing results directory (Line 155)."""
    client = TestClient(app)
    with patch("rtx_oom_guard.api.Path.exists", return_value=False):
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        assert response.json()["baseline"] == {}

def test_api_benchmarks_reads():
    """Verify benchmarks endpoint hits all read lines (Line 160-161)."""
    client = TestClient(app)
    with patch("rtx_oom_guard.api.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", side_effect=['{"b":1}', '{"d":1}']):
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        assert response.json()["baseline"] == {"b":1}
        assert response.json()["defrag"] == {"d":1}

def test_api_catch_all_file(tmp_path):
    """Verify SPA catch-all serves physical files (Line 181)."""
    # Create dummy dashboard dist
    dist = tmp_path / "dashboard" / "dist"
    dist.mkdir(parents=True)
    # Hitting a file that is NOT in /assets mount
    manifest = dist / "manifest.json"
    manifest.write_text("dummy manifest")
    
    # Correctly patch the DASHBOARD_PATH at the module level
    with patch("rtx_oom_guard.api.DASHBOARD_PATH", dist):
         client = TestClient(app)
         # Using manifest.json instead of assets/logo.png to reach catch_all
         response = client.get("/manifest.json")
         assert response.status_code == 200
         assert response.text == "dummy manifest"
