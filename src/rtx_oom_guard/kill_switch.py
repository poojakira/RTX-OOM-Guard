"""Kill switch with recovery mechanism — thread-safe, monotonic clock."""

import time
import threading
import logging

log = logging.getLogger("rtx_oom_guard.kill_switch")


class KillSwitch:
    """Circuit breaker for the defrag monitor with exponential backoff recovery.

    Uses monotonic clock (immune to NTP adjustments) and thread-safe state.
    """

    def __init__(
        self,
        max_latency_ms: float = 5.0,
        max_failures: int = 3,
        base_cooldown_s: float = 1.0,
        max_cooldown_s: float = 30.0,
    ):
        self.max_latency_ms = max_latency_ms
        self.max_failures = max_failures
        self.base_cooldown_s = base_cooldown_s
        self.max_cooldown_s = max_cooldown_s

        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._cooldown_until: float = 0.0
        self._permanently_disabled = False

    @property
    def is_active(self) -> bool:
        with self._lock:
            if self._permanently_disabled:
                return False
            if time.monotonic() < self._cooldown_until:
                return False
            return True

    def record_latency(self, latency_ms: float):
        with self._lock:
            if latency_ms <= self.max_latency_ms:
                if self._consecutive_failures > 0:
                    log.info(f"Kill switch recovered after {self._consecutive_failures} failures")
                self._consecutive_failures = 0
                return

            self._consecutive_failures += 1
            log.warning(
                f"Latency spike: {latency_ms:.1f}ms > {self.max_latency_ms}ms "
                f"(failure {self._consecutive_failures}/{self.max_failures})"
            )

            if self._consecutive_failures >= self.max_failures:
                self._permanently_disabled = True
                log.error("Kill switch: permanently disabled after max failures")
                return

            cooldown = min(
                self.base_cooldown_s * (2 ** (self._consecutive_failures - 1)),
                self.max_cooldown_s,
            )
            self._cooldown_until = time.monotonic() + cooldown
            log.info(f"Kill switch: cooling down for {cooldown:.1f}s")

    def reset(self):
        with self._lock:
            self._consecutive_failures = 0
            self._cooldown_until = 0.0
            self._permanently_disabled = False
            log.info("Kill switch manually reset")
