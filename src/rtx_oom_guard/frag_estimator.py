"""
EWMA-based fragmentation estimator — replaces the unused FragPredictor.

The FragPredictor Transformer had no training script, no dataset, and no
validated weights. This module provides a lightweight, validated alternative
using exponentially weighted moving average on real CUDA memory stats.
"""

import torch
import logging

log = logging.getLogger("rtx_oom_guard.frag_estimator")


class FragmentationEstimator:
    """Estimates GPU memory fragmentation using EWMA on allocation stats.

    Fragmentation ratio = 1 - (allocated / reserved)
    When allocated << reserved, memory is fragmented (lots of holes).
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 0.3):
        """
        Args:
            alpha: EWMA smoothing factor (higher = more responsive).
            threshold: Fragmentation ratio above which compaction is recommended.
        """
        self.alpha = alpha
        self.threshold = threshold
        self._ewma: float = 0.0
        self._samples: int = 0

    def update(self) -> float:
        """Sample current CUDA memory stats and update EWMA estimate.

        Returns:
            Current smoothed fragmentation ratio.
        """
        if not torch.cuda.is_available():
            return 0.0

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        if reserved == 0:
            return 0.0

        frag_ratio = 1.0 - (allocated / reserved)

        if self._samples == 0:
            self._ewma = frag_ratio
        else:
            self._ewma = self.alpha * frag_ratio + (1 - self.alpha) * self._ewma

        self._samples += 1
        return self._ewma

    @property
    def should_compact(self) -> bool:
        """Whether fragmentation exceeds threshold."""
        return self._ewma > self.threshold

    @property
    def current_estimate(self) -> float:
        return self._ewma

    def reset(self):
        self._ewma = 0.0
        self._samples = 0
