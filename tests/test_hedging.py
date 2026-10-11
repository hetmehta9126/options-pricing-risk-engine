"""
Tests for the delta-hedging simulation.

The key sanity check: with zero transaction costs and very frequent
rebalancing, discrete hedging should approximate continuous hedging, so
the spread (standard deviation) of the final P&L across paths should be
small, and much smaller than hedging the same option infrequently.
"""

from options_engine.black_scholes import BSInputs
from options_engine.hedging import run_hedging_simulation

INPUTS = BSInputs(spot=100, strike=100, maturity=1.0, rate=0.0557, volatility=0.20)


def test_zero_cost_frequent_rebalancing_gives_small_pnl_spread():
    result = run_hedging_simulation(
        inputs=INPUTS,
        num_paths=2_000,
        num_steps=252 * 4,  # a few rebalances per trading day
        rebalance_every=1,
        cost_rate=0.0,
        seed=42,
    )

    # Spread should be small relative to the ~10-11 price of the option itself.
    assert result.pnl.std() < 0.5


def test_pnl_spread_shrinks_as_rebalancing_gets_more_frequent():
    coarse = run_hedging_simulation(
        inputs=INPUTS, num_paths=2_000, num_steps=252, rebalance_every=21, cost_rate=0.0, seed=42,
    )
    fine = run_hedging_simulation(
        inputs=INPUTS, num_paths=2_000, num_steps=252, rebalance_every=1, cost_rate=0.0, seed=42,
    )

    assert fine.pnl.std() < coarse.pnl.std()


def test_pnl_spread_grows_with_transaction_costs():
    no_cost = run_hedging_simulation(
        inputs=INPUTS, num_paths=2_000, num_steps=252, rebalance_every=1, cost_rate=0.0, seed=42,
    )
    with_cost = run_hedging_simulation(
        inputs=INPUTS, num_paths=2_000, num_steps=252, rebalance_every=1, cost_rate=0.01, seed=42,
    )

    assert with_cost.pnl.std() >= no_cost.pnl.std()


def test_zero_cost_pnl_has_small_mean():
    """With zero costs, hedging a correctly-priced option should break even on average."""
    result = run_hedging_simulation(
        inputs=INPUTS, num_paths=5_000, num_steps=252, rebalance_every=1, cost_rate=0.0, seed=42,
    )
    assert abs(result.pnl.mean()) < 0.2
