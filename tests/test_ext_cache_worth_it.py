"""Own tests for Extension 3 — cache_is_worth_it() / cache_breakeven_reads()."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_breakeven_reads_matches_anthropic_style_numbers():
    # write premium 1.25x, read discount 0.10 -> breakeven = 1.25 / 0.9
    assert abs(pricing.cache_breakeven_reads(1.25, 0.10) - (1.25 / 0.9)) < 1e-9


def test_cache_worth_it_above_and_below_breakeven():
    breakeven = pricing.cache_breakeven_reads(1.25, 0.10)
    assert pricing.cache_is_worth_it(breakeven + 1, 1.25, 0.10) is True
    assert pricing.cache_is_worth_it(breakeven - 1, 1.25, 0.10) is False


def test_higher_write_cost_needs_more_reads():
    cheap_breakeven = pricing.cache_breakeven_reads(1.25, 0.10)
    pricey_breakeven = pricing.cache_breakeven_reads(2.0, 0.10)
    assert pricey_breakeven > cheap_breakeven
