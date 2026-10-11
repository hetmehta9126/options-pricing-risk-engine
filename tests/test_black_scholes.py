"""
Tests for the Black-Scholes-Merton pricer.

Includes a hand-checkable reference value and an independent
recomputation of the price straight from the d1/d2 formulas, so this
test does not just re-check the pricer against itself.
"""

import math

import pytest
from scipy.stats import norm

from options_engine.black_scholes import BSInputs, call_price, put_price, put_call_parity_difference


def test_known_reference_value():
    """S=100, K=100, T=1, r=5%, sigma=20% should give call ~10.45, put ~5.57."""
    inputs = BSInputs(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.20)

    call = call_price(inputs)
    put = put_price(inputs)

    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


def test_known_reference_value_independent_calculation():
    """Recompute the same price by hand from d1/d2, independently of call_price()."""
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_by_hand = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_by_hand = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    inputs = BSInputs(spot=S, strike=K, maturity=T, rate=r, volatility=sigma)
    assert call_price(inputs) == pytest.approx(call_by_hand, abs=1e-10)
    assert put_price(inputs) == pytest.approx(put_by_hand, abs=1e-10)


@pytest.mark.parametrize(
    "spot,strike,maturity,rate,vol,div",
    [
        (100, 100, 1.0, 0.0557, 0.20, 0.0),
        (120, 100, 0.5, 0.03, 0.35, 0.0),
        (80, 100, 2.0, 0.05, 0.25, 0.02),
        (100, 90, 0.25, 0.01, 0.15, 0.015),
    ],
)
def test_put_call_parity(spot, strike, maturity, rate, vol, div):
    """call - put should equal S*e^-qT - K*e^-rT for any valid inputs."""
    inputs = BSInputs(spot, strike, maturity, rate, vol, div)
    assert put_call_parity_difference(inputs) == pytest.approx(0.0, abs=1e-8)


def test_dividend_yield_lowers_call_raises_put():
    """A positive dividend yield should reduce the call price and raise the put price."""
    no_div = BSInputs(100, 100, 1, 0.05, 0.20, dividend_yield=0.0)
    with_div = BSInputs(100, 100, 1, 0.05, 0.20, dividend_yield=0.03)

    assert call_price(with_div) < call_price(no_div)
    assert put_price(with_div) > put_price(no_div)


def test_invalid_option_type_raises():
    from options_engine.black_scholes import price

    inputs = BSInputs(100, 100, 1, 0.05, 0.20)
    with pytest.raises(ValueError):
        price(inputs, "straddle")
