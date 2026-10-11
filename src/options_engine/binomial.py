"""
Cox-Ross-Rubinstein (CRR) binomial tree pricer for European options.

This is a discrete-time approximation to the same geometric Brownian
motion model that Black-Scholes prices in closed form. As the number of
steps grows, the binomial price should converge to the Black-Scholes
price for the same inputs. That convergence is checked in
tests/test_binomial.py and plotted in plots.py.

All prices here are model prices for a simulated instrument, not real
market quotes.
"""

import numpy as np

from .black_scholes import BSInputs


def crr_price(inputs: BSInputs, option_type: str, num_steps: int) -> float:
    """
    Price a European option with a CRR binomial tree.

    Because the option is European (no early exercise), we don't need to
    walk the tree node by node checking for early exercise. Instead we
    compute the terminal stock prices directly and discount the expected
    payoff back one step at a time using the risk-neutral probability.
    This is mathematically equivalent to the full tree walk but much
    faster in numpy.
    """
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.maturity,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    dt = T / num_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(
            "risk-neutral probability out of (0, 1); try more steps or check inputs"
        )

    # Terminal stock prices: S * u^j * d^(n-j) for j = 0..n up moves.
    j = np.arange(num_steps + 1)
    terminal_prices = S * (u ** j) * (d ** (num_steps - j))

    if option_type.lower() == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - terminal_prices, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    # Risk-neutral binomial probability mass at each terminal node.
    log_binom_coeff = (
        _log_factorial(num_steps) - _log_factorial(j) - _log_factorial(num_steps - j)
    )
    log_prob = log_binom_coeff + j * np.log(p) + (num_steps - j) * np.log(1.0 - p)
    probabilities = np.exp(log_prob)

    expected_payoff = np.sum(probabilities * payoffs)
    return np.exp(-r * T) * expected_payoff


def _log_factorial(n):
    """Log of n! via scipy's gammaln, vectorized, to avoid overflow for large n."""
    from scipy.special import gammaln

    return gammaln(np.asarray(n, dtype=float) + 1.0)
