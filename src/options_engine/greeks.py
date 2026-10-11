"""
Closed-form Black-Scholes-Merton Greeks for European options.

Units / conventions used in this module (these matter a lot for Greeks,
so they are spelled out here instead of just in the README):

- delta:  dPrice/dSpot, unitless (change in option price per 1 unit move
          in the underlying).
- gamma:  d^2Price/dSpot^2, i.e. d(delta)/dSpot. Same for calls and puts.
- vega:   dPrice/dVolatility, reported PER 1 VOLATILITY POINT (i.e. per
          0.01 change in sigma), which is the usual trading convention.
          The raw calculus derivative (per unit of sigma) is divided by
          100 to get this.
- theta:  dPrice/dTime, reported PER CALENDAR DAY, i.e. the raw calculus
          derivative with respect to time-to-maturity (which is per
          year) is negated and divided by 365.
- rho:    dPrice/dRate, reported PER 1 PERCENTAGE POINT change in the
          risk-free rate (i.e. per 0.01 change in r), same convention
          as vega.

All Greeks below are also checked against central finite differences in
tests/test_greeks.py.
"""

import numpy as np
from scipy.stats import norm

from .black_scholes import BSInputs, _d1_d2


def delta(inputs: BSInputs, option_type: str) -> float:
    """Delta: sensitivity of price to a 1-unit move in the underlying."""
    d1, _ = _d1_d2(inputs)
    q, T = inputs.dividend_yield, inputs.maturity
    if option_type.lower() == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    if option_type.lower() == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(inputs: BSInputs) -> float:
    """Gamma: d(delta)/dSpot. Identical for calls and puts."""
    S, T, sigma, q = inputs.spot, inputs.maturity, inputs.volatility, inputs.dividend_yield
    d1, _ = _d1_d2(inputs)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(inputs: BSInputs) -> float:
    """Vega: change in price per 1 volatility point (0.01 change in sigma)."""
    S, T, q = inputs.spot, inputs.maturity, inputs.dividend_yield
    d1, _ = _d1_d2(inputs)
    vega_per_unit_sigma = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return vega_per_unit_sigma / 100.0


def theta(inputs: BSInputs, option_type: str) -> float:
    """Theta: change in price per calendar day (time decay)."""
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.maturity,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    d1, d2 = _d1_d2(inputs)

    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type.lower() == "call":
        theta_per_year = (
            common
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
    elif option_type.lower() == "put":
        theta_per_year = (
            common
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return theta_per_year / 365.0


def rho(inputs: BSInputs, option_type: str) -> float:
    """Rho: change in price per 1 percentage point move in the risk-free rate."""
    K, T, r = inputs.strike, inputs.maturity, inputs.rate
    _, d2 = _d1_d2(inputs)

    if option_type.lower() == "call":
        rho_per_unit_rate = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        rho_per_unit_rate = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return rho_per_unit_rate / 100.0


def all_greeks(inputs: BSInputs, option_type: str) -> dict:
    """Convenience helper: return all five Greeks in one dict."""
    return {
        "delta": delta(inputs, option_type),
        "gamma": gamma(inputs),
        "vega": vega(inputs),
        "theta": theta(inputs, option_type),
        "rho": rho(inputs, option_type),
    }
