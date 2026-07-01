"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    old_policy_monthly = 0.0  # original flat policy: interrupt_rate=0.05, reserved always 3yr
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # --- original (pre-extension) policy, for the required before/after comparison ---
        old_tier = pricing.recommend_tier(hpd, interruptible)
        if old_tier == "spot":
            old_opt = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
        elif old_tier == "reserved":
            old_opt = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            old_opt = on_demand_cost
        old_policy_monthly += old_opt

        # --- extended policy: GPU-specific interrupt rate + job-duration-aware term ---
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=job_days)
        if tier == "spot":
            interrupt_rate = pricing.SPOT_INTERRUPT_RATE.get(gtype, 0.05)
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od, interrupt_rate=interrupt_rate)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            term = pricing.recommend_reserved_term(job_days, kind=j.get("kind"))
            price_col = "reserved_3yr_hr" if term == "3yr" else "reserved_1yr_hr"
            opt_cost = gpu_hours * num(c[price_col])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    old_savings_pct = (on_demand_monthly - old_policy_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"\n[Extension 1] policy comparison: old flat policy {old_savings_pct:.1f}% saved  ->  "
              f"new GPU-aware + duration-aware policy {savings_pct:.1f}% saved")
        print("[Extension 1] tier matrix (sample GPU types x duty cycle x interruptible):")
        matrix = pricing.tier_recommendation_matrix(["H100", "A10G", "L4"])
        for m in matrix:
            print(f"  {m['gpu_type']:7}duty={m['duty_cycle']:.2f}  interruptible={str(m['interruptible']):5}"
                  f"  interrupt_rate={m['spot_interrupt_rate']:.2f}  -> {m['tier']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "old_policy_savings_pct": round(old_savings_pct, 1)}


if __name__ == "__main__":
    run()
