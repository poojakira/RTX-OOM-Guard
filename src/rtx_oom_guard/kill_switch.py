"""
Kill switch with recovery mechanism.

Fixes the permanent kill switch issue — previously if prediction latency
exceeded 5ms, the monitor exited forever. Now uses a cooldown + retry
approach: backs off exponentially, retries after cooldown, only permanently
disables after max_failures consecutive failures.
"""

import time
import logging

log = logging.getLogger("rtx_oom_guard.kill_switch")


class KillSwitch:
    """Manages monitor health with recovery instead of permanent shutdown.

    After a latency spike:
    1. Enter cooldown (exponential backoff)
    2. After cooldown, retry
    3. Only permanently disable after max_failures consecutive failures
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

        self._consecutive_failures = 0
        self._cooldown_until: float = 0.0
        self._permanently_disabled = False

    @property
    def is_active(self) -> bool:
        """Whether the monitor should be running."""
        if self._permanently_disabled:
            return False
        if time.time() < self._cooldown_until:
            return False
        return True

    def record_latency(self, latency_ms: float):
        """Record a prediction latency measurement."""
        if latency_ms <= self.max_latency_ms:
            # Success — reset failure counter
            if self._consecutive_failures > 0:
                log.info(f"Kill switch recovered after {self._consecutive_failures} failures")
            self._consecutive_failures = 0
            return

        # Failure
        self._consecutive_failures += 1
        log.warning(
            f"Latency spike: {latency_ms:.1f}ms > {self.max_latency_ms}ms "
            f"(failure {self._consecutive_failures}/{self.max_failures})"
        )

        if self._consecutive_failures >= self.max_failures:
            self._permanently_disabled = True
            log.error("Kill switch: permanently disabled after max failures")
            return

        # Exponential backoff cooldown
        cooldown = min(
            self.base_cooldown_s * (2 ** (self._consecutive_failures - 1)),
            self.max_cooldown_s,
        )
        self._cooldown_until = time.time() + cooldown
        log.info(f"Kill switch: cooling down for {cooldown:.1f}s")

    def reset(self):
        """Manual reset — re-enables the monitor."""
        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        self._permanently_disabled = False
        log.info("Kill switch manually reset")
