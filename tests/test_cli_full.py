import pytest
from unittest.mock import patch
from rtx_oom_guard.cli import main

def test_cli_help():
    """Verify help output."""
    with patch("sys.stdout") as mock_stdout, \
         patch("sys.argv", ["rtx_oom_guard", "--help"]), \
         pytest.raises(SystemExit):
        main()

def test_cli_status():
    """Verify status command."""
    with patch("sys.argv", ["rtx_oom_guard", "status"]), \
         patch("rtx_oom_guard.cli._print") as mock_print:
        main()
        # Check that it printed something about health
        assert any("health" in str(call) or "READY" in str(call) for call in mock_print.call_args_list)

def test_cli_server_mocked():
    """Verify server command starts uvicorn."""
    with patch("sys.argv", ["rtx_oom_guard", "server", "--port", "8001"]), \
         patch("uvicorn.run") as mock_run:
        main()
        assert mock_run.called
        assert mock_run.call_args[1]['port'] == 8001

def test_cli_dashboard_mocked():
    """Verify dashboard command starts sync and loop."""
    # We mock time.sleep to avoid infinite loop
    with patch("sys.argv", ["rtx_oom_guard", "dashboard"]), \
         patch("rtx_oom_guard.dashboard.DashboardManager") as mock_mgr, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        main()
        assert mock_mgr.return_value.start_sync.called
        assert mock_mgr.return_value.start_dashboard.called

def test_cli_train_mocked(tmp_path):
    """Verify train command path."""
    with patch("sys.argv", ["rtx_oom_guard", "train", "--epochs", "1", "--trace-dir", str(tmp_path)]), \
         patch("rtx_oom_guard.trainer.trainer.train") as mock_train:
        main()
        assert mock_train.called

def test_cli_simulate_mocked():
    """Verify simulate command path."""
    with patch("sys.argv", ["rtx_oom_guard", "simulate", "--runs", "1"]), \
         patch("benchmarks.run_local_benchmark.main") as mock_bench:
        main()
        assert mock_bench.called

def test_cli_profile_mocked(tmp_path):
    """Verify profile command path."""
    # Note: cli.py 'profile' calls collect_from_model
    with patch("sys.argv", ["rtx_oom_guard", "profile", "--model", "gpt2", "--iterations", "1"]), \
         patch("rtx_oom_guard.profiler.collector.collect_from_model") as mock_collect:
        main()
        assert mock_collect.called
        assert mock_collect.call_args[0][0] == "gpt2"

def test_cli_mock_telemetry_loop():
    """Verify mock-telemetry command loop."""
    with patch("sys.argv", ["rtx_oom_guard", "mock-telemetry", "--interval", "0.1"]), \
         patch("rtx_oom_guard.defrag_engine.defragmenter.GPUMemoryDefragmenter._persist_telemetry") as mock_persist, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        main()
        assert mock_persist.called
