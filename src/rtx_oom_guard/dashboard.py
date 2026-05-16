"""
rtx_oom_guard.dashboard — Management utility for the AEON CORE monitoring dashboard.
"""

import os
import time
import subprocess
import threading
import shutil
from pathlib import Path
from rtx_oom_guard.utils import get_logger, DefragConfig  # type: ignore

log = get_logger("dashboard-mgr")

class DashboardManager:
    """
    Orchestrates the Vite dashboard and real-time telemetry syncing.
    """
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).absolute()
        self.dashboard_dir = self.root_dir / "dashboard"
        self.results_dir = self.root_dir / "results"
        self.public_live_dir = self.dashboard_dir / "public" / "live"
        self._stop_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._vite_proc: subprocess.Popen | None = None
        self.config = DefragConfig() # type: ignore

    def _ensure_dirs(self):
        """Ensure all required directories exist."""
        self.results_dir.mkdir(exist_ok=True)
        self.public_live_dir.mkdir(parents=True, exist_ok=True)

    def _sync_loop(self):
        """Background loop to sync JSON results to the dashboard public folder."""
        log.info("Telemetry sync active: [results/] -> [dashboard/public/live/]")

        files_to_sync = ["live_telemetry.json", "comparison.json", "baseline.json", "defrag.json"]

        while not self._stop_event.is_set():
            for filename in files_to_sync:
                src = self.results_dir / filename
                dst = self.public_live_dir / filename
                if src.exists():
                    try:
                        # Use atomic copy if possible
                        shutil.copy2(src, dst)
                    except Exception as e:
                        log.debug(f"Sync failed for {filename}: {e}")

            # Sync commands back (Manual Defrag button)
            cmd_src = self.public_live_dir / "commands.json"
            cmd_dst = self.results_dir / "commands.json"
            if cmd_src.exists():
                try:
                    shutil.copy2(cmd_src, cmd_dst)
                except Exception as e:
                    log.debug(f"Command sync failed: {e}")

            time.sleep(0.5)

    def start_dashboard(self):
        """Start the Vite dev server."""
        if not (self.dashboard_dir / "node_modules").exists():
            log.error("node_modules not found in dashboard/. Run 'npm install' in the dashboard directory first.")
            return

        log.info("Launching AEON CORE Dashboard via Vite...")
        try:
            # Use shell=True for Windows compatibility with npm
            self._vite_proc = subprocess.Popen(["npm", "run", "dev"], cwd=self.dashboard_dir, shell=True)
        except Exception as e:
            log.error(f"Failed to start Vite: {e}")

    def stop_dashboard(self):
        """Stop the Vite dev server."""
        if self._vite_proc:
            log.info("Stopping Vite dashboard...")
            # On Windows, taskkill is more reliable for shell=True processes
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self._vite_proc.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self._vite_proc.terminate()
            self._vite_proc = None

    def start_sync(self):
        """Start the background sync process."""
        self._ensure_dirs()
        self._stop_event.clear()
        thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread = thread
        thread.start()

    def stop_sync(self):
        """Stop the background sync process."""
        self._stop_event.set()
        thread = self._sync_thread
        if thread is not None and thread.is_alive() and threading.current_thread() != thread:
            try:
                thread.join(timeout=1.0)
            except RuntimeError:
                pass
        self._sync_thread = None

def main():
    """CLI entry point for the dashboard manager."""
    mgr = DashboardManager()
    mgr.start_sync()
    mgr.start_dashboard()

    log.info("Dashboard & Sync service running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        mgr.stop_sync()

if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
