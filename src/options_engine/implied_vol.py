"""
Implied volatility solver.

Given a market (or in our case, simulated) option price, find the
volatility that makes the Black-Scholes price match it. We use
Newton-Raphson first since it converges fast when vega isn't tiny, and
fall back to bisection if Newton-Raphson misbehaves (overshoots out of
bounds, stalls, or vega is too close to zero to trust the step).

Before solving, every price is checked against the no-arbitrage bounds
for a European option. A price outside those bounds has no valid
implied volatility and the solver returns that explicitly instead of
pretending to find an answer.
"""

from dataclasses import dataclass

import numpy as np

from . import config
from .black_scholes import BSInputs, price
from .greeks import vega as bs_vega


@dataclass
class IVResult:
    """Outcome of an implied volatility solve."""

    implied_vol: float | None   # None if no valid IV could be found
    iterations: int
    method: str                  # "newton", "bisection", or "out_of_bounds"
    converged: bool


def no_arbitrage_bounds(inputs: BSInputs, option_type: str) -> tuple[float, float]:
    """
    Return the (lower, upper) no-arbitrage price bounds for a European
    option, given everything except volatility.

    Call bounds: max(0, S*e^-qT - K*e^-rT) <= C <= S*e^-qT
    Put bounds:  max(0, K*e^-rT - S*e^-qT) <= P <= K*e^-rT
    """
    S, K, T, r, q = inputs.spot, inputs.strike, inputs.maturity, inputs.rate, inputs.dividend_yield
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type.lower() == "call":
        return max(0.0, discounted_spot - discounted_strike), discounted_spot
    if option_type.lower() == "put":
        return max(0.0, discounted_strike - discounted_spot), discounted_strike
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def implied_volatility(
    target_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> IVResult:
    """Solve for the volatility that reproduces target_price."""

    base_inputs = BSInputs(spot, strike, maturity, rate, config.IV_INITIAL_GUESS, dividend_yield)
    lower_bound, upper_bound = no_arbitrage_bounds(base_inputs, option_type)

    # A tiny tolerance so prices sitting exactly on the boundary (which
    # correspond to vol -> 0) aren't rejected by floating point noise.
    eps = 1e-10
    if target_price < lower_bound - eps or target_price > upper_bound + eps:
        return IVResult(implied_vol=None, iterations=0, method="out_of_bounds", converged=False)

    newton_result = _newton_raphson(target_price, spot, strike, maturity, rate, option_type, dividend_yield)
    if newton_result.converged:
        return newton_result

    return _bisection(target_price, spot, strike, maturity, rate, option_type, dividend_yield)


def _price_at(sigma, spot, strike, maturity, rate, option_type, dividend_yield) -> float:
    inputs = BSInputs(spot, strike, maturity, rate, sigma, dividend_yield)
    return price(inputs, option_type)


def _newton_raphson(target_price, spot, strike, maturity, rate, option_type, dividend_yield) -> IVResult:
    sigma = config.IV_INITIAL_GUESS
    low, high = config.IV_BOUNDS

    for i in range(1, config.IV_MAX_ITER_NEWTON + 1):
        inputs = BSInputs(spot, strike, maturity, rate, sigma, dividend_yield)
        model_price = price(inputs, option_type)
        diff = model_price - target_price

        if abs(diff) < config.IV_TOLERANCE:
            return IVResult(implied_vol=sigma, iterations=i, method="newton", converged=True)

        # vega from greeks.py is per 1 vol point (per 0.01 sigma), so we
        # convert back to a per-unit-sigma derivative for the Newton step.
        derivative = bs_vega(inputs) * 100.0

        if derivative < 1e-8:
            # Vega too flat to trust the step; let bisection take over.
            return IVResult(implied_vol=None, iterations=i, method="newton", converged=False)

        sigma = sigma - diff / derivative

        if sigma <= low or sigma >= high or not np.isfinite(sigma):
            return IVResult(implied_vol=None, iterations=i, method="newton", converged=False)

    return IVResult(implied_vol=None, iterations=config.IV_MAX_ITER_NEWTON, method="newton", converged=False)


def _bisection(target_price, spot, strike, maturity, rate, option_type, dividend_yield) -> IVResult:
    low, high = config.IV_BOUNDS

    f_low = _price_at(low, spot, strike, maturity, rate, option_type, dividend_yield) - target_price
    f_high = _price_at(high, spot, strike, maturity, rate, option_type, dividend_yield) - target_price

    if f_low * f_high > 0:
        # Target price not bracketed within our search bounds at all.
        return IVResult(implied_vol=None, iterations=0, method="bisection", converged=False)

    for i in range(1, config.IV_MAX_ITER_BISECTION + 1):
        mid = 0.5 * (low + high)
        f_mid = _price_at(mid, spot, strike, maturity, rate, option_type, dividend_yield) - target_price

        if abs(f_mid) < config.IV_TOLERANCE or (high - low) / 2 < config.IV_TOLERANCE:
            return IVResult(implied_vol=mid, iterations=i, method="bisection", converged=True)

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return IVResult(implied_vol=0.5 * (low + high), iterations=config.IV_MAX_ITER_BISECTION, method="bisection", converged=True)
