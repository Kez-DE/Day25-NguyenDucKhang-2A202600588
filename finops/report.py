"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 insights: list[str] | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI: GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
    if insights:
        lines += ["", "## Analysis", ""]
        lines += [f"- {line}" for line in insights]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def build_insights(r1: dict, r3: dict, levers: dict, sust: dict) -> list[str]:
    """Derive root-cause + prioritized-action + sustainability commentary from the
    missions' own result dicts (no hardcoded numbers — regenerates from whatever
    the current run produced).
    """
    out = []

    # --- root cause of each GPU-Util lie ---
    lie_ids = {l["gpu_id"] for l in r1["lies"]}
    for s in r1["summary"]:
        if s["gpu_id"] not in lie_ids:
            continue
        if s["mbu"] >= 0.5:
            mechanism = (f"MBU is high ({s['mbu']:.0%}) while MFU stays low ({s['mfu']:.0%}): "
                         "the SM is busy waiting on HBM bandwidth (memory-bound, e.g. decode), "
                         "not doing FLOPs, so nvidia-smi's util counter reads 'busy' the whole time.")
        else:
            mechanism = (f"Both MFU ({s['mfu']:.0%}) and MBU ({s['mbu']:.0%}) are low despite "
                         f"{s['gpu_util_pct']:.0f}% util. The SM is scheduled but idling on "
                         "synchronization/kernel-launch overhead (small batches, collective-comm "
                         "waits), which nvidia-smi still counts as 'active'.")
        out.append(f"**{s['gpu_id']}** ({s['gpu_type']}) is a GPU-Util lie: {mechanism}")

    # --- prioritized actions, ranked by $ magnitude ---
    ranked = sorted(levers.items(), key=lambda kv: kv[1], reverse=True)
    for i, (name, amount) in enumerate(ranked, start=1):
        out.append(f"Priority {i}: **{name}** (${amount:,.0f}/mo), "
                    f"{'highest ROI, act first' if i == 1 else 'next after higher-priority levers land'}.")

    # --- sustainability tied to $ cost ---
    best_region = sust.get("best_region", "n/a")
    base_region = sust.get("baseline_region", "n/a")
    base_cost, best_cost = sust.get("baseline_energy_cost_usd", 0), sust.get("best_region_energy_cost_usd", 0)
    base_carbon, best_carbon = sust.get("carbon_g", 0), sust.get("best_region_carbon_g", 0)
    if base_cost > 0:
        out.append(f"Per query, **{base_region}** costs ${base_cost:.6f} in electricity and "
                   f"emits {base_carbon:.3f} gCO2e; **{best_region}** costs ${best_cost:.6f} "
                   f"({(1 - best_cost / base_cost) * 100:.0f}% cheaper) and emits {best_carbon:.3f} "
                   f"gCO2e ({(1 - best_carbon / base_carbon) * 100:.0f}% less) for the same query. "
                   "Region choice cuts $ and carbon together rather than trading one for the other.")

    if r3.get("old_policy_savings_pct") is not None:
        out.append(f"Purchasing policy tuned for GPU-specific interruption rates and job duration "
                   f"moved savings from {r3['old_policy_savings_pct']:.1f}% (flat policy) to "
                   f"{r3['savings_pct']:.1f}% (GPU + duration aware policy).")

    return out


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
