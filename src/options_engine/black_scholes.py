"""
Black-Scholes-Merton pricing for European options.

Assumptions baked into this model (standard Black-Scholes-Merton):
- The underlying follows geometric Brownian motion with constant volatility.
- Constant, continuously compounded risk-free rate and dividend yield.
- European exercise only (no early exercise).
- No transaction costs or taxes in the pricing formula itself (the
  hedging simulation in hedging.py adds costs separately).

All prices coming out of this module are theoretical / model prices for a
simulated instrument. They are not quotes from any real market.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class BSInputs:
    """Container for the five Black-Scholes-Merton inputs."""

    spot: float          # S, current price of the underlying
    strike: float        # K, strike price
    maturity: float       # T, time to expiry in years
    rate: float           # r, continuously compounded risk-free rate
    volatility: float     # sigma, annualized volatility
    dividend_yield: float = 0.0  # q, continuous dividend yield


def _d1_d2(inputs: BSInputs) -> tuple[float, float]:
    """Compute the d1 and d2 terms shared by price and Greek formulas."""
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.maturity,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    if T <= 0 or sigma <= 0:
        raise ValueError("maturity and volatility must both be positive")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(inputs: BSInputs) -> float:
    """Price a European call under Black-Scholes-Merton."""
    S, K, T, r, q = inputs.spot, inputs.strike, inputs.maturity, inputs.rate, inputs.dividend_yield
    d1, d2 = _d1_d2(inputs)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(inputs: BSInputs) -> float:
    """Price a European put under Black-Scholes-Merton."""
    S, K, T, r, q = inputs.spot, inputs.strike, inputs.maturity, inputs.rate, inputs.dividend_yield
    d1, d2 = _d1_d2(inputs)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def price(inputs: BSInputs, option_type: str) -> float:
    """
    Price a European option of the given type.

    option_type must be "call" or "put".
    """
    option_type = option_type.lower()
    if option_type == "call":
        return call_price(inputs)
    if option_type == "put":
        return put_price(inputs)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def put_call_parity_difference(inputs: BSInputs) -> float:
    """
    Return call - put - (S * e^-qT - K * e^-rT).

    Under put-call parity this should be ~0. Useful as a sanity check
    on the pricing functions above.
    """
    S, K, T, r, q = inputs.spot, inputs.strike, inputs.maturity, inputs.rate, inputs.dividend_yield
    c = call_price(inputs)
    p = put_price(inputs)
    forward_value = S * np.exp(-q * T) - K * np.exp(-r * T)
    return c - p - forward_value
