import pytest
from unittest.mock import patch
from rtx_oom_guard.cli import main, collect_cmd, train_cmd, benchmark_cmd, serve_cmd, dashboard_cmd

# 1. Main Subcommand Branches
def test_cli_mock_telemetry_branch():
    """Verify mock-telemetry subcommand branch."""
    with patch("sys.argv", ["rtx_oom_guard", "mock-telemetry", "--interval", "0.01"]), \
         patch("rtx_oom_guard.defrag_engine.defragmenter.GPUMemoryDefragmenter") as mock_engine_cls, \
         patch("time.sleep", side_effect=[None, InterruptedError("stop")]):
        try:
            main()
        except InterruptedError:
            pass
        assert mock_engine_cls.called

def test_cli_status_branch():
    """Verify status subcommand branch."""
    with patch("sys.argv", ["rtx_oom_guard", "status"]), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Mock NVIDIA GPU"), \
         patch("rtx_oom_guard.utils.DefragConfig.load"), \
         patch("os.path.exists", return_value=True), \
         patch("pathlib.Path.exists", return_value=True):
        main()
        assert True

def test_cli_dashboard_branch_main():
    """Verify dashboard subcommand branch in main()."""
    with patch("sys.argv", ["rtx_oom_guard", "dashboard"]), \
         patch("rtx_oom_guard.dashboard.DashboardManager") as mock_mgr_cls:
        
        mock_mgr = mock_mgr_cls.return_value
        # Mock while loop to exit immediately
        with patch("time.sleep", side_effect=InterruptedError("stop")):
            try:
                main()
            except InterruptedError:
                pass
        assert mock_mgr.start_sync.called
        assert mock_mgr.start_dashboard.called

# 2. Individual Command Functions
def test_collect_cmd():
    """Verify individual collect_cmd entry point."""
    with patch("sys.argv", ["rtx_oom_guard-collect", "--model", "gpt2", "--iterations", "1"]), \
         patch("rtx_oom_guard.profiler.collector.collect_from_model") as mock_collect:
        collect_cmd()
        assert mock_collect.called

def test_train_cmd():
    """Verify individual train_cmd entry point."""
    with patch("sys.argv", ["rtx_oom_guard-train", "--epochs", "1"]), \
         patch("rtx_oom_guard.trainer.trainer.train") as mock_train:
        train_cmd()
        assert mock_train.called

def test_benchmark_cmd():
    """Verify individual benchmark_cmd entry point."""
    with patch("sys.argv", ["rtx_oom_guard-benchmark", "--runs", "1"]), \
         patch("benchmarks.run_local_benchmark.main") as mock_bench:
        benchmark_cmd()
        assert mock_bench.called

def test_serve_cmd():
    """Verify individual serve_cmd entry point."""
    with patch("sys.argv", ["rtx_oom_guard-serve"]), \
         patch("uvicorn.run") as mock_run:
        serve_cmd()
        assert mock_run.called

def test_dashboard_cmd_standalone():
    """Verify individual dashboard_cmd entry point."""
    with patch("sys.argv", ["rtx_oom_guard-dashboard"]), \
         patch("subprocess.Popen") as mock_popen, \
         patch("webbrowser.open"), \
         patch("time.sleep"):
        
        mock_proc = mock_popen.return_value
        mock_proc.poll.return_value = 0 # finished immediately
        dashboard_cmd()
        assert mock_popen.called

def test_cli_error_handling():
    """Verify CLI error handling for unknown command."""
    with patch("sys.argv", ["rtx_oom_guard", "unknown"]), \
         pytest.raises(SystemExit):
        main()
