"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def cache_breakeven_reads(write_cost_per_m: float, read_discount: float = 0.10) -> float:
    """Reads needed (at a normalized base price of 1.0/M-token) to earn back a cache write.

    Each cached read saves (1 - read_discount) vs. paying full price again; a write
    costs write_cost_per_m (typically a premium over base, e.g. 1.25x for a short-TTL
    cache). Break-even is where accumulated read savings equal that write premium:
    write_cost_per_m = N * (1 - read_discount)  ->  N = write_cost_per_m / (1 - read_discount)
    """
    return write_cost_per_m / max(1e-9, 1.0 - read_discount)


def cache_is_worth_it(avg_cache_reads: float, write_cost_per_m: float, read_discount: float = 0.10) -> bool:
    """True when a cached prefix's expected reuse earns back its write cost."""
    return avg_cache_reads > cache_breakeven_reads(write_cost_per_m, read_discount)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Spot reclamation rate by GPU type (illustrative 2026 neocloud data): flagship
# training cards are reclaimed less often than commodity inference cards, since
# providers keep more spare capacity on the SKUs everyone wants for training.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.02,
    "A100": 0.05, "MI300X": 0.05,
    "A10G": 0.12, "L4": 0.15,
}
SPOT_INTERRUPT_TOLERANCE = 0.15  # above this, checkpoint rework eats the discount

# `days` in workloads.csv is observed activity within a 30-day billing window, not
# job lifetime — so the reserved gate is a fraction of that window, not an absolute
# day count. Below half the window, even the shallower 1yr discount isn't worth it.
MIN_RESERVED_WINDOW_FRACTION = 0.5
MIN_DAYS_FOR_RESERVED = 30 * MIN_RESERVED_WINDOW_FRACTION
# A workload only earns the deeper 3yr discount if it's observed running near
# continuously across its whole billing window (a steady production service,
# not a short or bursty one) — trained jobs never qualify, they're finite projects.
RESERVED_3YR_WINDOW_FRACTION = 0.9


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45,
                    gpu_type: str | None = None, job_days: float | None = None) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Extended policy (instructor extension point):
      - interruptible & not 24/7 & the GPU's reclaim rate is tolerable -> 'spot'
        (a GPU type reclaimed too often burns the discount on checkpoint rework)
      - duty cycle >= break-even & the job runs long enough to earn back a
        committed term -> 'reserved'  (job_days=None preserves the original,
        term-agnostic behavior for callers that don't know the job's horizon)
      - otherwise -> 'on_demand'

    gpu_type/job_days are optional so existing call sites and tests keep their
    original behavior; see recommend_reserved_term() for the 1yr-vs-3yr call and
    tier_recommendation_matrix() for a full GPU x duty-cycle x interruptible sweep.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    interrupt_rate = SPOT_INTERRUPT_RATE.get(gpu_type, 0.05) if gpu_type else 0.05

    if interruptible and hours_per_day < 24 and interrupt_rate <= SPOT_INTERRUPT_TOLERANCE:
        return "spot"
    if duty >= be and (job_days is None or job_days >= MIN_DAYS_FOR_RESERVED):
        return "reserved"
    return "on_demand"


def recommend_reserved_term(job_days: float | None, kind: str | None = None,
                             window_days: float = 30) -> str:
    """Choose 1yr vs 3yr for a workload already routed to 'reserved'.

    Training runs are finite projects (they finish, get retrained, change
    architecture) — locking into 3yr risks paying for capacity the project won't
    need. Only a workload observed running near-continuously across the whole
    billing window (a steady inference service) earns the deeper 3yr discount.
    """
    if job_days is None:
        return "3yr"  # unknown horizon: keep the original, term-agnostic behavior
    if kind == "train":
        return "1yr"
    if job_days >= window_days * RESERVED_3YR_WINDOW_FRACTION:
        return "3yr"
    return "1yr"


def tier_recommendation_matrix(gpu_types: list[str], reserved_discount: float = 0.45,
                                duty_cycles: tuple[float, ...] = (0.15, 0.4, 0.55, 0.8, 1.0)) -> list[dict]:
    """Sweep GPU type x duty cycle x interruptible -> recommended tier.

    A reporting helper: shows how the policy above actually behaves across the
    decision space, not just for the specific jobs in workloads.csv.
    """
    rows = []
    for gpu_type in gpu_types:
        for duty in duty_cycles:
            hours_per_day = duty * 24.0
            for interruptible in (True, False):
                tier = recommend_tier(hours_per_day, interruptible, reserved_discount, gpu_type=gpu_type)
                rows.append({
                    "gpu_type": gpu_type, "duty_cycle": duty, "interruptible": interruptible,
                    "tier": tier, "spot_interrupt_rate": SPOT_INTERRUPT_RATE.get(gpu_type, 0.05),
                })
    return rows


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
