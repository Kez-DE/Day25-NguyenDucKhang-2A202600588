"""Own tests for Extension 1 — GPU-aware + duration-aware tier policy.

Does not modify tests/test_pricing.py; adds coverage for the new recommend_tier()
params, recommend_reserved_term(), and tier_recommendation_matrix().
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_recommend_tier_backward_compatible():
    # unchanged calls (no gpu_type/job_days) must keep the original behavior
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_high_interrupt_rate_gpu_avoids_spot():
    pricing.SPOT_INTERRUPT_RATE["consumer-gpu"] = 0.40  # far above tolerance
    try:
        tier = pricing.recommend_tier(2, True, gpu_type="consumer-gpu")
        assert tier != "spot"
    finally:
        del pricing.SPOT_INTERRUPT_RATE["consumer-gpu"]


def test_short_job_does_not_get_reserved():
    # high duty cycle, but the job only runs a handful of days -> stays on_demand
    tier = pricing.recommend_tier(24, False, job_days=5)
    assert tier == "on_demand"


def test_recommend_reserved_term():
    assert pricing.recommend_reserved_term(job_days=10, kind="train") == "1yr"      # train never gets 3yr
    assert pricing.recommend_reserved_term(job_days=30, kind="infer") == "3yr"      # full window -> 3yr
    assert pricing.recommend_reserved_term(job_days=10, kind="infer") == "1yr"      # partial window -> 1yr
    assert pricing.recommend_reserved_term(job_days=None) == "3yr"                  # unknown horizon: prior behavior


def test_tier_recommendation_matrix_shape():
    rows = pricing.tier_recommendation_matrix(["H100", "A10G"])
    assert len(rows) == 2 * 5 * 2  # gpu_types x duty_cycles x interruptible
    assert all(r["tier"] in {"spot", "reserved", "on_demand"} for r in rows)
