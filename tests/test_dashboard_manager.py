from unittest.mock import patch, MagicMock
from rtx_oom_guard.dashboard import DashboardManager

def test_dashboard_manager_ensure_dirs(tmp_path):
    """Verify directory creation is handled correctly."""
    mgr = DashboardManager(root_dir=str(tmp_path))
    mgr._ensure_dirs()
    assert (tmp_path / "results").exists()
    assert (tmp_path / "dashboard" / "public" / "live").exists()

def test_dashboard_manager_sync_loop_stops(tmp_path):
    """Verify that the sync loop copies files before exiting."""
    mgr = DashboardManager(root_dir=str(tmp_path))
    mgr._ensure_dirs()
    
    # Create a source file
    src = tmp_path / "results" / "live_telemetry.json"
    src.write_text("{}")
    
    with patch("time.sleep") as mock_sleep:
        # Side effect: Let it run once, check for the file, THEN set stop event
        def side_effect(arg):
            if (tmp_path / "dashboard" / "public" / "live" / "live_telemetry.json").exists():
                mgr._stop_event.set()
        mock_sleep.side_effect = side_effect
        
        mgr.start_sync()
        
        # Give it a bit of real time to ensure the thread can run
        import time as real_time
        max_wait = 2.0
        start_time = real_time.time()
        success = False
        while real_time.time() - start_time < max_wait:
            if (tmp_path / "dashboard" / "public" / "live" / "live_telemetry.json").exists():
                success = True
                # Trigger the stop event now that we've verified success
                mgr._stop_event.set()
                break
            real_time.sleep(0.05)
        
        assert success, "Telemetry file was not synced to dashboard public folder."
        mgr.stop_sync()

def test_dashboard_manager_vite_start_stop(tmp_path):
    """Verify Vite starting and stopping with correct process management."""
    mgr = DashboardManager(root_dir=str(tmp_path))
    (tmp_path / "dashboard" / "node_modules").mkdir(parents=True)
    
    with patch("subprocess.Popen") as mock_popen, \
         patch("sys.platform", "linux"):
        
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_popen.return_value = mock_proc
        
        mgr.start_dashboard()
        mock_popen.assert_called_once()
        assert "npm" in str(mock_popen.call_args[0][0])
        
        # Stop
        with patch("os.name", "posix"):
            mgr.stop_dashboard()
            mock_proc.terminate.assert_called_once()

    # Test Windows taskkill specifically
    with patch("subprocess.Popen") as mock_popen, \
         patch("os.name", "nt"), \
         patch("subprocess.run") as mock_run:
        
        mock_proc = MagicMock()
        mock_proc.pid = 5555
        mock_popen.return_value = mock_proc
        
        mgr.start_dashboard()
        mgr.stop_dashboard()
        mock_run.assert_called_once()
        assert "taskkill" in str(mock_run.call_args[0][0])
