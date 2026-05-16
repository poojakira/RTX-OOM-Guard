import json
from unittest.mock import patch, mock_open, MagicMock
from fastapi.testclient import TestClient
from rtx_oom_guard.api import app

client = TestClient(app)

def test_get_telemetry_retry_loop_on_json_error():
    """Verify that the API retries on a JSON decoding error exactly 3 times."""
    # Use builtins.open to catch the call within the api.py module
    with patch("rtx_oom_guard.api.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="INVALID_JSON")), \
         patch("time.sleep") as mock_sleep:
        
        response = client.get("/api/telemetry")
        assert response.status_code == 200
        data = response.json()
        assert data["total_compactions"] == 0
        # The retry loop is for _ in range(3), so sleep should be called 3 times
        assert mock_sleep.call_count == 3

def test_get_telemetry_success_after_retry():
    """Verify that the API succeeds if it fails once and then gets valid data."""
    # Custom iterator for file reads to simulate transient failure
    mock_file_content = [
        "INVALID_JSON", 
        json.dumps({"total_compactions": 10, "total_freed_mb": 500.0, "avg_latency_ms": 2.0})
    ]
    
    class MockFile:
        def __init__(self, contents):
            self.contents = contents
            self.counter = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            res = self.contents[self.counter]
            self.counter += 1
            return res

    mock_f = MockFile(mock_file_content)

    with patch("rtx_oom_guard.api.Path.exists", return_value=True), \
         patch("builtins.open", return_value=mock_f), \
         patch("time.sleep") as mock_sleep:
        
        response = client.get("/api/telemetry")
        assert response.status_code == 200
        data = response.json()
        assert data["total_compactions"] == 10
        assert mock_sleep.call_count == 1

def test_get_benchmarks_failure_handling():
    """Verify 500 return when results directory access fails."""
    # Patch read_text instead of json.loads to avoid breaking httpx's internal json parsing
    with patch("rtx_oom_guard.api.Path.exists", return_value=True), \
         patch("rtx_oom_guard.api.Path.read_text", side_effect=Exception("Disk failure")):
        
        response = client.get("/api/benchmarks")
        assert response.status_code == 500
        # Now it is safe to call .json() because the patch on Path.read_text doesn't affect httpx
        assert "Disk failure" in response.json()["detail"]

def test_gpu_snapshot_cuda_unavailable():
    """Verify memory API when CUDA is not present."""
    with patch("torch.cuda.is_available", return_value=False):
        # We need to ensure the import succeeds but is_available is False
        response = client.get("/api/memory")
        assert response.status_code == 200
        data = response.json()
        assert data["cuda_available"] == False

def test_catch_all_spa():
    """Verify the SPA catch-all route returns index.html for unknown paths."""
    # Patch at the class level to avoid WindowsPath attribute restrictions
    from pathlib import Path
    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "is_file", return_value=False), \
         patch("rtx_oom_guard.api.FileResponse") as mock_file_resp:
        
        mock_file_resp.return_value = MagicMock(status_code=200)
        response = client.get("/some-random-ui-route")
        assert response.status_code == 200
        # Verify it attempted to serve index.html
        args, _ = mock_file_resp.call_args
        assert "index.html" in str(args[0])
