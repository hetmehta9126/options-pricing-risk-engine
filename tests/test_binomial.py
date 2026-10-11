"""
Tests for the CRR binomial pricer: it should converge to Black-Scholes
as the number of steps grows, for both calls and puts.
"""

import pytest

from options_engine.binomial import crr_price
from options_engine.black_scholes import BSInputs, call_price, put_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_converges_to_black_scholes_with_many_steps(option_type):
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)
    bs_ref = call_price(inputs) if option_type == "call" else put_price(inputs)

    binom_price = crr_price(inputs, option_type, num_steps=1000)

    assert binom_price == pytest.approx(bs_ref, abs=0.01)


def test_error_shrinks_as_steps_increase():
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)
    bs_ref = call_price(inputs)

    error_few_steps = abs(crr_price(inputs, "call", num_steps=10) - bs_ref)
    error_many_steps = abs(crr_price(inputs, "call", num_steps=500) - bs_ref)

    assert error_many_steps < error_few_steps


def test_rejects_zero_steps():
    inputs = BSInputs(100, 100, 1.0, 0.0557, 0.20)
    with pytest.raises(ValueError):
        crr_price(inputs, "call", num_steps=0)
