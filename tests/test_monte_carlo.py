"""
Tests for the Monte Carlo pricer: with enough paths and a fixed seed,
the estimate should land within a few standard errors of the
Black-Scholes price.
"""

import pytest

from options_engine.monte_carlo import mc_price
from options_engine.black_scholes import BSInputs, call_price, put_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_price_within_a_few_std_errors_of_black_scholes(option_type):
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)
    bs_ref = call_price(inputs) if option_type == "call" else put_price(inputs)

    result = mc_price(inputs, option_type, num_paths=200_000, seed=42)

    assert abs(result.price - bs_ref) < 4 * result.std_error


def test_std_error_shrinks_with_more_paths():
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)

    small = mc_price(inputs, "call", num_paths=500, seed=42)
    large = mc_price(inputs, "call", num_paths=50_000, seed=42)

    assert large.std_error < small.std_error


def test_same_seed_gives_same_price():
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)

    first = mc_price(inputs, "call", num_paths=1_000, seed=7)
    second = mc_price(inputs, "call", num_paths=1_000, seed=7)

    assert first.price == second.price
