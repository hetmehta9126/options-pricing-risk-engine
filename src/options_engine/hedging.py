"""
Delta hedging simulation.

Story: we sell (go short) one European call for the Black-Scholes price,
then try to hedge that short position by holding shares of the
underlying equal to the option's delta, rebalancing at some fixed
frequency (daily, weekly, monthly, ...), with proportional transaction
costs on every trade. We do this over many simulated price paths and
look at the distribution of the final hedging P&L.

If we could rebalance continuously with no transaction costs, the P&L
would be exactly zero on every path (perfect replication). In practice
we rebalance discretely, so there's hedging error, and that error grows
as we rebalance less often. Transaction costs add a separate drag that
grows with how often we trade. This file lets us see both effects.

Everything here is simulated. There is no real trading, no real P&L,
and nothing here should be read as a claim about trading performance.
"""

from dataclasses import dataclass

import numpy as np

from .black_scholes import BSInputs
from .black_scholes import price as bs_price
from .greeks import delta as bs_delta


@dataclass
class HedgeSimResult:
    """Summary of a delta-hedging simulation across many paths."""

    pnl: np.ndarray            # final hedging P&L, one value per simulated path
    rebalance_every: int        # steps between rebalances
    cost_rate: float            # proportional transaction cost rate
    num_paths: int
    num_steps: int


def simulate_gbm_paths(
    inputs: BSInputs,
    num_paths: int,
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate num_paths risk-neutral GBM paths for the underlying, each
    with num_steps time steps from 0 to maturity.

    Returns an array of shape (num_paths, num_steps + 1), including the
    starting price S0 in column 0.
    """
    S, T, r, sigma, q = inputs.spot, inputs.maturity, inputs.rate, inputs.volatility, inputs.dividend_yield
    dt = T / num_steps

    z = rng.standard_normal((num_paths, num_steps))
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    log_paths = np.cumsum(log_returns, axis=1)
    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(log_paths)
    return paths


def run_hedging_simulation(
    inputs: BSInputs,
    num_paths: int,
    num_steps: int,
    rebalance_every: int,
    cost_rate: float,
    seed: int,
    option_type: str = "call",
) -> HedgeSimResult:
    """
    Run the short-option delta-hedging simulation described at the top
    of this file and return the final P&L for every simulated path.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(inputs, num_paths, num_steps, rng)

    T, r, sigma, q, K = inputs.maturity, inputs.rate, inputs.volatility, inputs.dividend_yield, inputs.strike
    dt = T / num_steps

    # Premium received for selling the option at t=0, using the same
    # Black-Scholes price we hedge against.
    premium = bs_price(inputs, option_type)

    S0 = paths[:, 0]
    delta_0 = bs_delta(inputs, option_type)
    shares = np.full(num_paths, delta_0)
    cash = premium - shares * S0 - cost_rate * np.abs(shares) * S0

    for i in range(1, num_steps + 1):
        S_i = paths[:, i]
        cash = cash * np.exp(r * dt)

        is_last_step = i == num_steps
        is_rebalance_step = (i % rebalance_every == 0) and not is_last_step

        if is_last_step:
            # Close out the hedge entirely at maturity.
            target_shares = np.zeros(num_paths)
        elif is_rebalance_step:
            remaining_maturity = T - i * dt
            hedge_inputs = BSInputs(S_i, K, remaining_maturity, r, sigma, q)
            target_shares = bs_delta(hedge_inputs, option_type)
        else:
            target_shares = shares

        trade = target_shares - shares
        cash = cash - trade * S_i - cost_rate * np.abs(trade) * S_i
        shares = target_shares

    payoff = np.maximum(paths[:, -1] - K, 0.0) if option_type.lower() == "call" else np.maximum(K - paths[:, -1], 0.0)
    final_pnl = cash - payoff

    return HedgeSimResult(
        pnl=final_pnl,
        rebalance_every=rebalance_every,
        cost_rate=cost_rate,
        num_paths=num_paths,
        num_steps=num_steps,
    )
