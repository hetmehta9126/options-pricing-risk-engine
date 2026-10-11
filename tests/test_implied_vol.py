"""
Tests for the implied volatility solver: it should recover the vol used
to generate a price, and it should correctly flag prices that fall
outside the no-arbitrage bounds instead of returning a fake answer.
"""

import pytest

from options_engine.black_scholes import BSInputs, call_price, put_price
from options_engine.implied_vol import implied_volatility, no_arbitrage_bounds


@pytest.mark.parametrize(
    "spot,strike,maturity,rate,vol,option_type",
    [
        (100, 100, 1.0, 0.0557, 0.20, "call"),
        (100, 100, 1.0, 0.0557, 0.20, "put"),
        (120, 100, 0.5, 0.03, 0.45, "call"),
        (80, 100, 2.0, 0.05, 0.10, "put"),
        (100, 100, 0.05, 0.0557, 0.60, "call"),  # short maturity, high vol
    ],
)
def test_recovers_input_volatility(spot, strike, maturity, rate, vol, option_type):
    inputs = BSInputs(spot, strike, maturity, rate, vol)
    target_price = call_price(inputs) if option_type == "call" else put_price(inputs)

    result = implied_volatility(target_price, spot, strike, maturity, rate, option_type)

    assert result.converged
    assert result.implied_vol == pytest.approx(vol, abs=1e-5)


def test_price_above_upper_bound_is_out_of_bounds():
    spot, strike, maturity, rate = 100, 100, 1.0, 0.0557
    _, upper = no_arbitrage_bounds(BSInputs(spot, strike, maturity, rate, 0.2), "call")

    result = implied_volatility(upper + 10.0, spot, strike, maturity, rate, "call")

    assert result.implied_vol is None
    assert result.method == "out_of_bounds"
    assert not result.converged


def test_price_below_lower_bound_is_out_of_bounds():
    spot, strike, maturity, rate = 100, 100, 1.0, 0.0557
    lower, _ = no_arbitrage_bounds(BSInputs(spot, strike, maturity, rate, 0.2), "call")

    result = implied_volatility(lower - 1.0, spot, strike, maturity, rate, "call")

    assert result.implied_vol is None
    assert result.method == "out_of_bounds"
    assert not result.converged


def test_no_arbitrage_bounds_are_sane():
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)
    call_lower, call_upper = no_arbitrage_bounds(inputs, "call")
    put_lower, put_upper = no_arbitrage_bounds(inputs, "put")

    assert 0.0 <= call_lower < call_upper
    assert 0.0 <= put_lower < put_upper
    # The actual Black-Scholes price should sit strictly inside its bounds.
    assert call_lower < call_price(inputs) < call_upper
    assert put_lower < put_price(inputs) < put_upper
