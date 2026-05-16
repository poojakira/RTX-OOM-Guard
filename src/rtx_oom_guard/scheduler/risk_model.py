"""
rtx_oom_guard.scheduler.risk_model — Simple OOM-risk scorer.

Two modes:
  1. Rule-based (default) — weighted heuristic over fragmentation,
     memory utilisation, and allocation rate-of-change.
  2. Logistic regression — sklearn wrapper for labelled history.

No deep learning.  All scores land in [0, 1].

Usage::

    model = OOMRiskModel()
    score = model.score(fragmentation=0.72, utilisation=0.91, alloc_delta_mb=15.0)
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

log = logging.getLogger("rtx_oom_guard.risk_model")


@dataclass
class RiskThresholds:
    """Tunable knobs for the rule-based scorer."""
    frag_weight: float = 0.45
    util_weight: float = 0.35
    delta_weight: float = 0.20
    frag_critical: float = 0.70     # fragmentation above this is concerning
    util_critical: float = 0.90     # utilisation above this is concerning
    delta_scale_mb: float = 50.0    # normalise alloc-delta to this MB range


class OOMRiskModel:
    """
    Lightweight OOM-risk estimator.

    Parameters
    ----------
    mode : str
        ``"rule"`` for the heuristic scorer, ``"logistic"`` for sklearn LR.
    thresholds : RiskThresholds, optional
        Custom weights / knobs for rule mode.
    """

    def __init__(
        self,
        mode: str = "rule",
        thresholds: Optional[RiskThresholds] = None,
    ) -> None:
        if mode not in ("rule", "logistic"):
            raise ValueError(f"Unknown mode '{mode}'; choose 'rule' or 'logistic'")
        self.mode = mode
        self.thresholds = thresholds or RiskThresholds()
        self._lr_model: Any = None       # sklearn LogisticRegression (lazy)
        self._history: List[Dict[str, float]] = []

    # -- Rule-based scorer -------------------------------------------------

    def _rule_score(
        self,
        fragmentation: float,
        utilisation: float,
        alloc_delta_mb: float,
    ) -> float:
        t = self.thresholds
        frag_norm = min(fragmentation / max(t.frag_critical, 1e-9), 1.0)
        util_norm = min(utilisation / max(t.util_critical, 1e-9), 1.0)
        delta_norm = min(abs(alloc_delta_mb) / max(t.delta_scale_mb, 1e-9), 1.0)

        raw = (
            t.frag_weight * frag_norm
            + t.util_weight * util_norm
            + t.delta_weight * delta_norm
        )
        # Sigmoid squash to keep in (0, 1) and smooth the boundary
        return 1.0 / (1.0 + math.exp(-6.0 * (raw - 0.5)))

    # -- Logistic regression scorer ----------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OOMRiskModel":
        """
        Fit the logistic-regression model on labelled data.

        Parameters
        ----------
        X : (n_samples, 3) array
            Columns: fragmentation, utilisation, alloc_delta_mb.
        y : (n_samples,) binary array
            1 = OOM observed, 0 = no OOM.
        """
        from sklearn.linear_model import LogisticRegression  # lazy import

        self._lr_model = LogisticRegression(max_iter=500, solver="lbfgs")
        self._lr_model.fit(X, y)
        self.mode = "logistic"
        log.info("Logistic model fitted on %d samples", len(y))
        return self

    def _logistic_score(
        self,
        fragmentation: float,
        utilisation: float,
        alloc_delta_mb: float,
    ) -> float:
        if self._lr_model is None:
            raise RuntimeError("Logistic model not fitted — call .fit() first")
        x = np.array([[fragmentation, utilisation, alloc_delta_mb]])
        proba = self._lr_model.predict_proba(x)
        # probability of class 1 (OOM)
        return float(proba[0, 1])

    # -- Unified API -------------------------------------------------------

    def score(
        self,
        fragmentation: float = 0.0,
        utilisation: float = 0.0,
        alloc_delta_mb: float = 0.0,
    ) -> float:
        """
        Return an OOM-risk score in [0, 1].

        Parameters
        ----------
        fragmentation : float
            Current fragmentation ratio (0 = none, 1 = fully fragmented).
        utilisation : float
            GPU memory utilisation (allocated / total), in [0, 1].
        alloc_delta_mb : float
            Recent change in allocated memory (MB). Positive = growing.
        """
        if self.mode == "logistic":
            s = self._logistic_score(fragmentation, utilisation, alloc_delta_mb)
        else:
            s = self._rule_score(fragmentation, utilisation, alloc_delta_mb)

        entry = {
            "fragmentation": fragmentation,
            "utilisation": utilisation,
            "alloc_delta_mb": alloc_delta_mb,
            "risk_score": round(s, 6),
        }
        self._history.append(entry)
        return round(s, 6)

    @property
    def history(self) -> List[Dict[str, float]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
