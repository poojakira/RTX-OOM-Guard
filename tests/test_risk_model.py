"""
tests/test_risk_model.py — Prove the OOMRiskModel actually runs.
"""

import pytest
import numpy as np

from rtx_oom_guard.scheduler.risk_model import OOMRiskModel, RiskThresholds


class TestRuleBasedScorer:
    """Rule-based OOM-risk model tests."""

    def test_score_range(self):
        """Score is always in [0, 1]."""
        model = OOMRiskModel(mode="rule")
        for _ in range(100):
            frag = np.random.uniform(0, 1)
            util = np.random.uniform(0, 1)
            delta = np.random.uniform(-100, 100)
            score = model.score(frag, util, delta)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_high_frag_high_score(self):
        """High fragmentation and utilisation → high risk."""
        model = OOMRiskModel(mode="rule")
        score = model.score(fragmentation=0.95, utilisation=0.95, alloc_delta_mb=40.0)
        assert score > 0.7, f"Expected high-risk score, got {score}"

    def test_low_frag_low_score(self):
        """Low fragmentation and utilisation → low risk."""
        model = OOMRiskModel(mode="rule")
        score = model.score(fragmentation=0.05, utilisation=0.20, alloc_delta_mb=0.0)
        assert score < 0.3, f"Expected low-risk score, got {score}"

    def test_monotonic_fragmentation(self):
        """Higher fragmentation → higher score (all else equal)."""
        model = OOMRiskModel(mode="rule")
        s_low = model.score(fragmentation=0.1, utilisation=0.5, alloc_delta_mb=0.0)
        s_high = model.score(fragmentation=0.9, utilisation=0.5, alloc_delta_mb=0.0)
        assert s_high > s_low

    def test_history_recorded(self):
        """Each score call appends to history."""
        model = OOMRiskModel(mode="rule")
        model.score(0.1, 0.2, 0.0)
        model.score(0.3, 0.4, 5.0)
        assert len(model.history) == 2
        assert "risk_score" in model.history[0]

    def test_clear_history(self):
        model = OOMRiskModel(mode="rule")
        model.score(0.5, 0.5, 0.0)
        model.clear_history()
        assert len(model.history) == 0

    def test_custom_thresholds(self):
        """Custom RiskThresholds change the output."""
        default = OOMRiskModel(mode="rule")
        custom = OOMRiskModel(
            mode="rule",
            thresholds=RiskThresholds(frag_weight=0.9, util_weight=0.05, delta_weight=0.05),
        )
        s_default = default.score(0.8, 0.3, 0.0)
        s_custom = custom.score(0.8, 0.3, 0.0)
        # Custom heavily weights fragmentation, so it should differ
        assert s_default != s_custom


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestLogisticScorer:
    """Logistic-regression OOM-risk model tests."""

    def test_fit_predict(self):
        """fit + score works on synthetic data."""
        rng = np.random.RandomState(42)
        X = rng.rand(200, 3)
        # Label: OOM when all features are high
        y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5) & (X[:, 2] > 0.3)).astype(int)

        model = OOMRiskModel(mode="rule")
        model.fit(X, y)
        assert model.mode == "logistic"

        score = model.score(0.9, 0.9, 40.0)
        assert 0.0 <= score <= 1.0

    def test_logistic_not_fitted_raises(self):
        """Calling logistic score without fit raises RuntimeError."""
        model = OOMRiskModel(mode="logistic")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.score(0.5, 0.5, 0.0)

    def test_invalid_mode(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            OOMRiskModel(mode="deep_neural_net")
