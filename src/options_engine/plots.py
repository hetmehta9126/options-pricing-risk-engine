"""
Chart-making functions for the pricing and risk engine.

Every chart in this module is built from simulated or synthetic
numbers (model prices, simulated paths, a toy volatility skew). None of
it is real market data, and that is called out again in each chart's
title so a chart can never be mistaken for a real trading result.

Colors are picked from a fixed categorical palette so colors mean the
same thing across charts, and are not just whatever matplotlib's
default cycle happens to give us.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical palette (same hues used consistently across charts).
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_VIOLET = "#4a3aa7"

COLOR_SURFACE = "#fcfcfb"
COLOR_PRIMARY_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"

FREQUENCY_COLORS = {
    "daily": COLOR_BLUE,
    "weekly": COLOR_ORANGE,
    "monthly": COLOR_AQUA,
}


def _style_axes(ax):
    """Apply a consistent, low-contrast look to an axes object."""
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BASELINE)
    ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_SECONDARY_INK)
    ax.xaxis.label.set_color(COLOR_PRIMARY_INK)
    ax.yaxis.label.set_color(COLOR_PRIMARY_INK)
    ax.title.set_color(COLOR_PRIMARY_INK)


def _save(fig, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.patch.set_facecolor(COLOR_SURFACE)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_price_vs_spot(spot_grid, call_prices, put_prices, output_dir: str) -> str:
    """Line chart of simulated Black-Scholes call/put price vs spot price."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(ax)

    ax.plot(spot_grid, call_prices, color=COLOR_BLUE, linewidth=2, label="Call")
    ax.plot(spot_grid, put_prices, color=COLOR_ORANGE, linewidth=2, label="Put")

    ax.set_xlabel("Spot price (simulated)")
    ax.set_ylabel("Option price (model, simulated)")
    ax.set_title("Black-Scholes price vs spot -- simulated inputs, not market data")
    ax.legend(frameon=False)

    return _save(fig, output_dir, "price_vs_spot.png")


def plot_greeks_vs_spot(spot_grid, call_greeks: dict, put_greeks: dict, output_dir: str) -> str:
    """Small multiples of each Greek vs spot, call and put overlaid."""
    greek_names = ["delta", "gamma", "vega", "theta", "rho"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, name in zip(axes, greek_names):
        _style_axes(ax)
        ax.plot(spot_grid, call_greeks[name], color=COLOR_BLUE, linewidth=2, label="Call")
        ax.plot(spot_grid, put_greeks[name], color=COLOR_ORANGE, linewidth=2, label="Put")
        ax.set_title(name.capitalize())
        ax.set_xlabel("Spot price (simulated)")

    axes[0].legend(frameon=False)
    fig.suptitle("Greeks vs spot -- simulated inputs, not market data", color=COLOR_PRIMARY_INK)

    return _save(fig, output_dir, "greeks_vs_spot.png")


def plot_binomial_convergence(steps_grid, abs_errors, output_dir: str) -> str:
    """Log-log plot of |binomial price - Black-Scholes price| vs number of steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(ax)

    ax.plot(steps_grid, abs_errors, color=COLOR_BLUE, linewidth=2, marker="o", markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of binomial steps")
    ax.set_ylabel("Absolute error vs Black-Scholes (simulated)")
    ax.set_title("CRR binomial convergence -- simulated prices, not market data")

    return _save(fig, output_dir, "binomial_convergence.png")


def plot_mc_convergence(paths_grid, abs_errors, std_errors, output_dir: str) -> str:
    """Log-log plot of |MC price - Black-Scholes price| vs number of paths, with std error bars."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(ax)

    ax.errorbar(
        paths_grid,
        abs_errors,
        yerr=std_errors,
        color=COLOR_BLUE,
        ecolor=COLOR_MUTED,
        linewidth=2,
        marker="o",
        markersize=5,
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Absolute error vs Black-Scholes (simulated)")
    ax.set_title("Monte Carlo convergence -- simulated prices, not market data")

    return _save(fig, output_dir, "monte_carlo_convergence.png")


def plot_iv_smile(strikes, true_vols, recovered_vols, output_dir: str) -> str:
    """Synthetic skew curve vs the volatility recovered by the IV solver."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axes(ax)

    ax.plot(strikes, true_vols, color=COLOR_MUTED, linewidth=2, linestyle="--", label="Synthetic input skew")
    ax.plot(strikes, recovered_vols, color=COLOR_BLUE, linewidth=0, marker="o", markersize=5, label="Recovered by IV solver")

    ax.set_xlabel("Strike (synthetic)")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Implied volatility smile -- synthetic skew model, not real quotes")
    ax.legend(frameon=False)

    return _save(fig, output_dir, "iv_smile.png")


def plot_hedging_pnl_distributions(pnl_by_frequency_and_cost: dict, output_dir: str) -> str:
    """
    Grid of P&L histograms: one column per transaction cost rate, one
    overlaid histogram per rebalancing frequency.

    pnl_by_frequency_and_cost maps (frequency_label, cost_rate) -> pnl array.
    """
    cost_rates = sorted({cost for (_, cost) in pnl_by_frequency_and_cost.keys()})
    frequencies = list(FREQUENCY_COLORS.keys())

    fig, axes = plt.subplots(1, len(cost_rates), figsize=(6 * len(cost_rates), 4.5), sharex=True, sharey=True)
    if len(cost_rates) == 1:
        axes = [axes]

    for ax, cost_rate in zip(axes, cost_rates):
        _style_axes(ax)
        series_for_cost = [
            pnl_by_frequency_and_cost[(freq_label, cost_rate)]
            for freq_label in frequencies
            if (freq_label, cost_rate) in pnl_by_frequency_and_cost
        ]
        all_pnl = np.concatenate(series_for_cost)
        shared_bins = np.linspace(all_pnl.min(), all_pnl.max(), 41)

        for freq_label in frequencies:
            key = (freq_label, cost_rate)
            if key not in pnl_by_frequency_and_cost:
                continue
            pnl = pnl_by_frequency_and_cost[key]
            ax.hist(
                pnl,
                bins=shared_bins,
                color=FREQUENCY_COLORS[freq_label],
                alpha=0.55,
                label=freq_label,
                edgecolor="none",
            )
        ax.axvline(0.0, color=COLOR_BASELINE, linewidth=1)
        ax.set_title(f"Transaction cost rate = {cost_rate:.2%}")
        ax.set_xlabel("Hedging P&L (simulated)")

    axes[0].set_ylabel("Number of simulated paths")
    axes[0].legend(frameon=False)
    fig.suptitle("Delta-hedging P&L distribution -- simulated paths, not real trading results", color=COLOR_PRIMARY_INK)

    return _save(fig, output_dir, "hedging_pnl_distributions.png")
