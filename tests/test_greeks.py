"""
Tests for the closed-form Greeks, checked against central finite
differences of the Black-Scholes price. This also doubles as a check
that the unit conventions documented in greeks.py (vega per vol point,
theta per day, rho per 1% rate) are applied consistently.
"""

import pytest

from options_engine.black_scholes import BSInputs, price
from options_engine.greeks import delta, gamma, vega, theta, rho

SCENARIOS = [
    (100, 100, 1.0, 0.0557, 0.20, 0.0, "call"),
    (100, 100, 1.0, 0.0557, 0.20, 0.0, "put"),
    (120, 100, 0.5, 0.03, 0.35, 0.01, "call"),
    (80, 100, 2.0, 0.05, 0.25, 0.02, "put"),
]


@pytest.mark.parametrize("S,K,T,r,sigma,q,option_type", SCENARIOS)
def test_delta_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-4
    up = price(BSInputs(S + h, K, T, r, sigma, q), option_type)
    down = price(BSInputs(S - h, K, T, r, sigma, q), option_type)
    fd_delta = (up - down) / (2 * h)

    assert delta(BSInputs(S, K, T, r, sigma, q), option_type) == pytest.approx(fd_delta, abs=1e-3)


@pytest.mark.parametrize("S,K,T,r,sigma,q,option_type", SCENARIOS)
def test_gamma_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-2
    up = price(BSInputs(S + h, K, T, r, sigma, q), option_type)
    mid = price(BSInputs(S, K, T, r, sigma, q), option_type)
    down = price(BSInputs(S - h, K, T, r, sigma, q), option_type)
    fd_gamma = (up - 2 * mid + down) / (h**2)

    assert gamma(BSInputs(S, K, T, r, sigma, q)) == pytest.approx(fd_gamma, abs=1e-3)


@pytest.mark.parametrize("S,K,T,r,sigma,q,option_type", SCENARIOS)
def test_vega_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-4
    up = price(BSInputs(S, K, T, r, sigma + h, q), option_type)
    down = price(BSInputs(S, K, T, r, sigma - h, q), option_type)
    # fd derivative is per unit sigma; vega() is per 1 vol point (0.01 sigma).
    fd_vega_per_point = ((up - down) / (2 * h)) / 100.0

    assert vega(BSInputs(S, K, T, r, sigma, q)) == pytest.approx(fd_vega_per_point, abs=1e-3)


@pytest.mark.parametrize("S,K,T,r,sigma,q,option_type", SCENARIOS)
def test_theta_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-5
    up = price(BSInputs(S, K, T + h, r, sigma, q), option_type)
    down = price(BSInputs(S, K, T - h, r, sigma, q), option_type)
    # Price rises with more time to maturity, but theta measures decay as
    # time PASSES (T shrinks), so the sign is flipped. theta() is per day.
    fd_theta_per_day = -((up - down) / (2 * h)) / 365.0

    assert theta(BSInputs(S, K, T, r, sigma, q), option_type) == pytest.approx(fd_theta_per_day, abs=1e-3)


@pytest.mark.parametrize("S,K,T,r,sigma,q,option_type", SCENARIOS)
def test_rho_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-4
    up = price(BSInputs(S, K, T, r + h, sigma, q), option_type)
    down = price(BSInputs(S, K, T, r - h, sigma, q), option_type)
    fd_rho_per_point = ((up - down) / (2 * h)) / 100.0

    assert rho(BSInputs(S, K, T, r, sigma, q), option_type) == pytest.approx(fd_rho_per_point, abs=1e-3)


def test_call_delta_between_zero_and_one():
    inputs = BSInputs(100, 100, 1, 0.0557, 0.20)
    assert 0.0 < delta(inputs, "call") < 1.0


def test_put_delta_between_minus_one_and_zero():
    inputs = BSInputs(100, 100, 1, 0.0557, 0.20)
    assert -1.0 < delta(inputs, "put") < 0.0
