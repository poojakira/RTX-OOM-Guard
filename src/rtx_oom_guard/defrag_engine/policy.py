"""
rtx_oom_guard.defrag_engine.policy — Honest OOM-risk mitigation policy.

Actions based on risk-score thresholds:

  +-----------+------+---------------------------------------------+
  | Risk      | Tier | Action                                      |
  +-----------+------+---------------------------------------------+
  | 0.0 – 0.5 | SAFE | no-op                                       |
  | 0.5 – 0.8 | WARN | log warning, suggest batch-size reduction    |
  | 0.8 – 1.0 | ACT  | empty_cache, suggest batch-size downshift    |
  +-----------+------+---------------------------------------------+

No magic — just cache-clearing and logged advice.

Usage::

    policy = MitigationPolicy()
    action = policy.evaluate(risk_score=0.85)
    # action.tier == "ACT", action.cache_cleared == True
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterable

log = logging.getLogger("rtx_oom_guard.mitigation_policy")


@dataclass
class MitigationAction:
    """Record of one policy evaluation."""
    timestamp: float
    risk_score: float
    tier: str                     # SAFE / WARN / ACT
    message: str
    mode: str = "PREDICTIVE"       # PREDICTIVE / REACTIVE
    cache_cleared: bool = False
    suggested_batch_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MitigationPolicy:
    """
    Threshold-driven mitigation policy.

    Parameters
    ----------
    warn_threshold : float
        Risk score above which a warning is logged.
    act_threshold : float
        Risk score above which active intervention (cache clear) occurs.
    batch_downshift_factor : float
        Multiply current batch size by this when suggesting a downshift.
    """

    def __init__(
        self,
        warn_threshold: float = 0.5,
        act_threshold: float = 0.8,
        batch_downshift_factor: float = 0.75,
        engine: Optional[Any] = None,
    ) -> None:
        self.warn_threshold = warn_threshold
        self.act_threshold = act_threshold
        self.batch_downshift_factor = batch_downshift_factor
        self.engine = engine
        self._actions: List[MitigationAction] = []

    def evaluate(
        self,
        risk_score: float,
        current_batch_size: int = 0,
        tensors_to_defragment: Optional[Iterable[Any]] = None,
        force_act: bool = False,
        mode: str = "PREDICTIVE",
    ) -> MitigationAction:
        """
        Evaluate the policy for a given risk score.

        Returns
        -------
        MitigationAction
            What was done (or not done).
        """
        ts = time.time()

        if risk_score >= self.act_threshold or force_act:
            suggested_bs = max(1, int(current_batch_size * self.batch_downshift_factor)) if current_batch_size else None

            risk_label = f"PEER-INDUCED COMPACTION (Local Risk: {risk_score:.3f})" if force_act and risk_score < self.act_threshold else f"HIGH RISK ({risk_score:.3f})"
            msg = (
                f"{risk_label} — cleared CUDA cache"
                + (f", suggest batch_size → {suggested_bs}" if suggested_bs else "")
            )

            # Active Defragmentation if engine is provided
            cache_cleared = False
            if self.engine is not None and tensors_to_defragment is not None:
                record = self.engine.defragment_tensors(tensors_to_defragment, reason="policy_act")
                cache_cleared = not record.get("skipped", False)
                if cache_cleared:
                    msg = f"{risk_label} — Defragmented {record.get('tensors_compacted', 0)} tensors, freed {record.get('freed_mb', 0):.1f} MB. " + (f"Suggest batch_size → {suggested_bs}" if suggested_bs else "")
            else:
                cache_cleared = self._try_empty_cache()

            action = MitigationAction(
                timestamp=ts,
                risk_score=risk_score,
                tier="ACT" if not force_act else "PEER_ACT",
                message=msg,
                mode=mode,
                cache_cleared=cache_cleared,
                suggested_batch_size=suggested_bs,
            )
            log.warning(msg)

        elif risk_score >= self.warn_threshold:
            suggested_bs = max(1, int(current_batch_size * self.batch_downshift_factor)) if current_batch_size else None
            msg = (
                f"ELEVATED RISK ({risk_score:.3f}) — consider reducing batch size"
                + (f" to {suggested_bs}" if suggested_bs else "")
            )
            action = MitigationAction(
                timestamp=ts,
                risk_score=risk_score,
                tier="WARN",
                message=msg,
                suggested_batch_size=suggested_bs,
            )
            log.warning(msg)

        else:
            action = MitigationAction(
                timestamp=ts,
                risk_score=risk_score,
                tier="SAFE",
                message="No action needed",
            )

        self._actions.append(action)

        # Trigger heartbeat telemetry for the dashboard
        if self.engine is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.engine._persist_telemetry(
                        torch.cuda.memory_allocated() / 1024**2,
                        torch.cuda.memory_reserved() / 1024**2
                    )
            except Exception:
                pass

        return action

    @property
    def actions(self) -> List[MitigationAction]:
        return list(self._actions)

    @property
    def action_counts(self) -> Dict[str, int]:
        counts = {"SAFE": 0, "WARN": 0, "ACT": 0, "PEER_ACT": 0}
        for a in self._actions:
            counts[a.tier] = counts.get(a.tier, 0) + 1
        return counts

    def clear(self) -> None:
        self._actions.clear()

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _try_empty_cache() -> bool:
        """Attempt torch.cuda.empty_cache(); return True if successful."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return True
        except Exception:
            pass
        return False
