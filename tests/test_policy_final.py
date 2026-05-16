from unittest.mock import MagicMock, patch
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy

def test_mitigation_policy_scenarios():
    """Verify MitigationPolicy triggers correct actions for risk tiers."""
    policy = MitigationPolicy(warn_threshold=0.4, act_threshold=0.7)
    
    # SAFE
    action = policy.evaluate(0.2)
    assert action.tier == "SAFE"
    assert not action.cache_cleared
    
    # WARN
    with patch("rtx_oom_guard.defrag_engine.policy.log") as mock_log:
        action = policy.evaluate(0.5)
        assert action.tier == "WARN"
        assert mock_log.warning.called
    
    # ACT (without engine)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        action = policy.evaluate(0.8)
        assert action.tier == "ACT"
        assert action.cache_cleared
        assert mock_empty.called

def test_mitigation_policy_with_engine():
    """Verify MitigationPolicy uses engine.defragment_tensors when provided."""
    mock_engine = MagicMock()
    mock_engine.defragment_tensors.return_value = {"freed_mb": 10.0, "tensors_compacted": 5}
    
    policy = MitigationPolicy(engine=mock_engine)
    tensors = [MagicMock()]
    
    with patch("torch.cuda.is_available", return_value=True):
        action = policy.evaluate(0.9, tensors_to_defragment=tensors)
        assert action.tier == "ACT"
        assert action.cache_cleared
        assert mock_engine.defragment_tensors.called
        assert "Defragmented 5 tensors" in action.message

def test_mitigation_policy_stats_and_clear():
    """Verify MitigationPolicy tracks action history."""
    policy = MitigationPolicy()
    policy.evaluate(0.1) # SAFE
    policy.evaluate(0.9) # ACT
    
    counts = policy.action_counts
    assert counts["SAFE"] == 1
    assert counts["ACT"] == 1
    
    assert len(policy.actions) == 2
    policy.clear()
    assert len(policy.actions) == 0

def test_mitigation_policy_force_act():
    """Verify force_act bypasses thresholds."""
    policy = MitigationPolicy(act_threshold=0.9)
    action = policy.evaluate(0.1, force_act=True)
    assert action.tier == "PEER_ACT"
    assert "PEER-INDUCED" in action.message
