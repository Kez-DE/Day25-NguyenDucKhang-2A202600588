"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

CACHE_DISCOUNT = 0.10        # cached-read price = 10% of base (matches request_cost default)
CACHE_WRITE_PREMIUM = 1.25   # writing a cache costs ~1.25x the base price (short-TTL cache)


def _avg_cache_reads_by_tier(rows: list[dict]) -> dict:
    """Proxy for how many times a cached prefix is reused, per model tier.

    token_usage.csv has no session/prefix id, so we approximate reuse by grouping
    cache-hit requests by (team, project): repeated requests from the same project
    are the most likely source of a shared, reused system-prompt/prefix.
    """
    groups = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            groups[r["route_tier"]][(r["team"], r["project"])] += 1
    return {tier: (sum(c.values()) / len(c) if c else 0.0) for tier, c in groups.items()}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    avg_cache_reads = _avg_cache_reads_by_tier(rows)
    breakeven_reads = pricing.cache_breakeven_reads(CACHE_WRITE_PREMIUM, CACHE_DISCOUNT)
    cache_worth_it = {
        tier: pricing.cache_is_worth_it(avg_cache_reads.get(tier, 0.0), CACHE_WRITE_PREMIUM, CACHE_DISCOUNT)
        for tier in MODEL_PRICES
    }

    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (only if it earns back its
        # write cost for this tier), batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cached_in = cached if cache_worth_it.get(r["route_tier"], True) else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached_in, batch=is_batch,
                                          cache_discount=CACHE_DISCOUNT)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"\n[Extension 3] cache break-even: {breakeven_reads:.2f} reads "
              f"(write premium {CACHE_WRITE_PREMIUM}x, read discount {CACHE_DISCOUNT})")
        for tier in MODEL_PRICES:
            reads = avg_cache_reads.get(tier, 0.0)
            print(f"[Extension 3] tier={tier:6}avg_cache_reads={reads:.1f}  "
                  f"worth_it={cache_worth_it[tier]}  ({reads / breakeven_reads:.1f}x break-even)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_breakeven_reads": round(breakeven_reads, 2),
        "avg_cache_reads": {k: round(v, 1) for k, v in avg_cache_reads.items()},
        "cache_worth_it": cache_worth_it,
    }


if __name__ == "__main__":
    run()
