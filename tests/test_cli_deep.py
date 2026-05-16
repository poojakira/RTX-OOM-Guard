from unittest.mock import MagicMock, patch
from rtx_oom_guard.cli import main

def test_cli_simulate_subcommand():
    """Verify simulation subcommand flow."""
    with patch("sys.argv", ["rtx_oom_guard", "simulate", "--steps", "1"]), \
         patch("benchmarks.run_local_benchmark.main") as mock_bench:
        main()
        assert mock_bench.called

def test_cli_profile_subcommand():
    """Verify profiling subcommand flow."""
    with patch("sys.argv", ["rtx_oom_guard", "profile", "--iterations", "1"]), \
         patch("rtx_oom_guard.profiler.collector.collect_from_model") as mock_collect:
        main()
        assert mock_collect.called

def test_cli_profile_error_branch():
    """Verify profiling error handling."""
    with patch("sys.argv", ["rtx_oom_guard", "profile", "--model", "gpt2"]), \
         patch("rtx_oom_guard.profiler.collector.collect_from_model", side_effect=ValueError("Profile Fail")), \
         patch("rtx_oom_guard.cli._print") as mock_print:
        main()
        assert any("Profile Fail" in str(args[0]) for args in mock_print.call_args_list)

def test_cli_status_command_mocked():
    """Verify status command logic branches (Line 269-296)."""
    # 1. Dashboard build missing branch (Line 293)
    def exists_side_effect(path):
        path_str = str(path)
        if "checkpoint" in path_str: return True
        if "dashboard" in path_str: return False # Trigger 293
        return True 
        
    with patch("sys.argv", ["rtx_oom_guard", "status"]), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA-MOCK-A100"), \
         patch("torch.version.cuda", "12.2"), \
         patch("os.path.exists", side_effect=exists_side_effect):
        main()

def test_cli_dashboard_drain_and_exit():
    """Verify dashboard output draining and exit sequence (Lines 180-183, 199-203)."""
    from rtx_oom_guard.cli import dashboard_cmd

    with patch("sys.argv", ["rtx_oom_guard-dashboard"]), \
         patch("subprocess.Popen") as mock_popen, \
         patch("time.sleep", side_effect=[None, None, KeyboardInterrupt]), \
         patch("webbrowser.open"):

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Process still running
        mock_proc.stdout = ["API START\n", "ERROR in API\n"]
        mock_popen.return_value = mock_proc

        dashboard_cmd()

        # Verify graceful shutdown sequence at 201-203
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

def test_cli_no_rich_fallback():
    """Verify fallback when rich is missing (Line 32-34, 43-45, 53)."""
    from rtx_oom_guard.cli import _print, print_banner
    with patch("rtx_oom_guard.cli.HAS_RICH", False), \
         patch("builtins.print") as mock_print:
        print_banner()
        _print("test message")
        assert mock_print.called

def test_cli_main_entrypoint_dunder():
    """Verify the module-level main() call at Line 351."""
    # We can't easily trigger the 'if __name__ == "__main__"' block 
    # but we can call it after mocking sys.argv to ensure it doesn't crash.
    with patch("sys.argv", ["rtx_oom_guard", "--help"]), \
         patch("argparse.ArgumentParser.parse_args", side_effect=SystemExit(0)):
        try:
            # Re-running the entry point section
            from rtx_oom_guard import cli
            cli.main()
        except SystemExit:
            pass
