from rtx_oom_guard.defrag_engine.policy import MitigationPolicy

def test_policy_safe_threshold():
    policy = MitigationPolicy(warn_threshold=0.5, act_threshold=0.8)
    action = policy.evaluate(risk_score=0.2, current_batch_size=32)
    assert action.tier == "SAFE"
    assert action.cache_cleared is False

def test_policy_warn_threshold():
    policy = MitigationPolicy(warn_threshold=0.5, act_threshold=0.8)
    action = policy.evaluate(risk_score=0.6, current_batch_size=32)
    assert action.tier == "WARN"
    assert action.suggested_batch_size == 24  # 32 * 0.75
    assert action.cache_cleared is False

def test_policy_act_threshold():
    policy = MitigationPolicy(warn_threshold=0.5, act_threshold=0.8)
    action = policy.evaluate(risk_score=0.9, current_batch_size=32)
    assert action.tier == "ACT"
    # CPU fallback just checks try_empty_cache which returns False if CUDA disabled

def test_policy_action_counts():
    policy = MitigationPolicy()
    policy.evaluate(risk_score=0.1)
    policy.evaluate(risk_score=0.6)
    policy.evaluate(risk_score=0.6)
    policy.evaluate(risk_score=0.9)
    counts = policy.action_counts
    assert counts["SAFE"] == 1
    assert counts["WARN"] == 2
    assert counts["ACT"] == 1

    assert len(policy.actions) == 4
    policy.clear()
    assert len(policy.actions) == 0
