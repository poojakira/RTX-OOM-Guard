from unittest.mock import patch, MagicMock
from rtx_oom_guard.dashboard import DashboardManager

def test_dashboard_manager_ensure_dirs(tmp_path):
    """Verify directory creation is handled correctly."""
    mgr = DashboardManager(root_dir=str(tmp_path))
    mgr._ensure_dirs()
    assert (tmp_path / "results").exists()
    assert (tmp_path / "dashboard" / "public" / "live").exists()

def test_dashboard_manager_vite_start_failure(tmp_path):
    """Verify error handling when Vite fails to start (Line 72-73)."""
    mgr = DashboardManager(root_dir=str(tmp_path))
    (tmp_path / "dashboard" / "node_modules").mkdir(parents=True)
    
    with patch("subprocess.Popen", side_effect=Exception("Spawn Fail")), \
         patch("rtx_oom_guard.dashboard.log") as mock_log:
        mgr.start_dashboard()
        assert mock_log.error.called

def test_dashboard_manager_stop_sync_runtime_error(tmp_path):
    """Verify runtime error handling during sync stop (Line 102-103)."""
    mgr = DashboardManager(root_dir=str(tmp_path))
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    mock_thread.join.side_effect = RuntimeError("Syncing thread cannot be joined")
    mgr._sync_thread = mock_thread
    
    # This should not raise except
    mgr.stop_sync()
    assert mock_thread.join.called

def test_dashboard_main_entrypoint():
    """Verify dashboard main entrypoint (Line 121)."""
    with patch("rtx_oom_guard.dashboard.DashboardManager") as mock_mgr_class, \
         patch("rtx_oom_guard.dashboard.log") as mock_log, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        from rtx_oom_guard.dashboard import main
        main()
        assert mock_mgr_class.return_value.start_sync.called
        assert mock_mgr_class.return_value.stop_sync.called
