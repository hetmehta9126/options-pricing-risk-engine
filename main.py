"""
Runs the full options pricing and risk engine pipeline end to end:

1. Price a base call/put scenario with Black-Scholes and compute Greeks.
2. Check put-call parity.
3. Solve for implied volatility on a known price (sanity check).
4. Compare the CRR binomial tree and Monte Carlo to Black-Scholes as
   steps/paths grow (convergence).
5. Build a synthetic volatility smile and recover it with the IV solver.
6. Run the delta-hedging simulation across rebalancing frequencies and
   transaction cost rates.
7. Save all charts and a Markdown report to outputs/.

Every number produced here is from a model or a simulation with fixed,
documented assumptions (see src/options_engine/config.py). None of it
is real market data.
"""

import os

import numpy as np

from src.options_engine import config, plots, report
from src.options_engine.binomial import crr_price
from src.options_engine.black_scholes import BSInputs, call_price, put_price, put_call_parity_difference
from src.options_engine.greeks import all_greeks
from src.options_engine.hedging import run_hedging_simulation
from src.options_engine.implied_vol import implied_volatility
from src.options_engine.monte_carlo import mc_price


def run_base_scenario():
    inputs = BSInputs(
        spot=config.DEFAULT_SPOT,
        strike=config.DEFAULT_STRIKE,
        maturity=config.DEFAULT_MATURITY,
        rate=config.RISK_FREE_RATE,
        volatility=config.DEFAULT_VOLATILITY,
        dividend_yield=config.DEFAULT_DIVIDEND_YIELD,
    )
    c_price = call_price(inputs)
    p_price = put_price(inputs)

    return {
        "spot": inputs.spot,
        "strike": inputs.strike,
        "maturity": inputs.maturity,
        "rate": inputs.rate,
        "volatility": inputs.volatility,
        "dividend_yield": inputs.dividend_yield,
        "call_price": c_price,
        "put_price": p_price,
        "call_greeks": all_greeks(inputs, "call"),
        "put_greeks": all_greeks(inputs, "put"),
        "parity_diff": put_call_parity_difference(inputs),
    }, inputs


def run_implied_vol_check(inputs: BSInputs):
    target_price = call_price(inputs)
    result = implied_volatility(
        target_price=target_price,
        spot=inputs.spot,
        strike=inputs.strike,
        maturity=inputs.maturity,
        rate=inputs.rate,
        option_type="call",
        dividend_yield=inputs.dividend_yield,
    )
    return {
        "input_vol": inputs.volatility,
        "recovered_vol": result.implied_vol,
        "method": result.method,
        "iterations": result.iterations,
    }


def run_binomial_convergence(inputs: BSInputs):
    bs_ref = call_price(inputs)
    steps_list = config.BINOMIAL_STEPS_GRID
    prices = [crr_price(inputs, "call", n) for n in steps_list]
    abs_errors = [abs(p - bs_ref) for p in prices]
    return {"steps": steps_list, "prices": prices, "abs_errors": abs_errors}


def run_mc_convergence(inputs: BSInputs):
    bs_ref = call_price(inputs)
    paths_list = config.MC_PATHS_GRID
    prices, std_errors, abs_errors = [], [], []
    for n in paths_list:
        result = mc_price(inputs, "call", num_paths=n, seed=config.RANDOM_SEED)
        prices.append(result.price)
        std_errors.append(result.std_error)
        abs_errors.append(abs(result.price - bs_ref))
    return {"paths": paths_list, "prices": prices, "std_errors": std_errors, "abs_errors": abs_errors}


def run_iv_smile(inputs: BSInputs):
    strikes = np.linspace(config.SMILE_STRIKE_GRID_MIN, config.SMILE_STRIKE_GRID_MAX, config.SMILE_STRIKE_GRID_POINTS)
    moneyness = strikes / inputs.spot - 1.0
    true_vols = (
        config.SMILE_BASE_VOL
        + config.SMILE_SKEW_SLOPE * moneyness
        + config.SMILE_CURVATURE * moneyness**2
    )

    recovered_vols = []
    for strike, vol in zip(strikes, true_vols):
        scenario = BSInputs(inputs.spot, strike, inputs.maturity, inputs.rate, vol, inputs.dividend_yield)
        synthetic_price = call_price(scenario)
        result = implied_volatility(
            target_price=synthetic_price,
            spot=inputs.spot,
            strike=strike,
            maturity=inputs.maturity,
            rate=inputs.rate,
            option_type="call",
            dividend_yield=inputs.dividend_yield,
        )
        recovered_vols.append(result.implied_vol)

    recovered_vols = np.array(recovered_vols)
    max_abs_error = float(np.max(np.abs(recovered_vols - true_vols)))

    return {
        "strikes": strikes,
        "true_vols": true_vols,
        "recovered_vols": recovered_vols,
        "max_abs_error": max_abs_error,
    }


def run_hedging_study(inputs: BSInputs):
    num_steps = int(round(config.HEDGE_NUM_STEPS_PER_YEAR * inputs.maturity))
    pnl_by_frequency_and_cost = {}
    summary_rows = []

    for freq_label, rebalance_every in config.HEDGE_REBALANCE_FREQUENCIES.items():
        for cost_rate in config.HEDGE_TRANSACTION_COST_RATES:
            result = run_hedging_simulation(
                inputs=inputs,
                num_paths=config.HEDGE_NUM_PATHS,
                num_steps=num_steps,
                rebalance_every=rebalance_every,
                cost_rate=cost_rate,
                seed=config.RANDOM_SEED,
                option_type="call",
            )
            pnl_by_frequency_and_cost[(freq_label, cost_rate)] = result.pnl
            summary_rows.append(
                {
                    "frequency": freq_label,
                    "cost_rate": cost_rate,
                    "mean_pnl": float(np.mean(result.pnl)),
                    "std_pnl": float(np.std(result.pnl)),
                }
            )

    return pnl_by_frequency_and_cost, {
        "num_paths": config.HEDGE_NUM_PATHS,
        "num_steps": num_steps,
        "maturity": inputs.maturity,
        "seed": config.RANDOM_SEED,
        "rows": summary_rows,
    }


def main():
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    base_scenario, base_inputs = run_base_scenario()
    iv_check = run_implied_vol_check(base_inputs)
    binom_convergence = run_binomial_convergence(base_inputs)
    mc_convergence = run_mc_convergence(base_inputs)
    smile = run_iv_smile(base_inputs)
    pnl_by_key, hedging_summary = run_hedging_study(base_inputs)

    spot_grid = np.linspace(config.SPOT_GRID_MIN, config.SPOT_GRID_MAX, config.SPOT_GRID_POINTS)
    call_prices, put_prices = [], []
    call_greeks_grid = {k: [] for k in ["delta", "gamma", "vega", "theta", "rho"]}
    put_greeks_grid = {k: [] for k in ["delta", "gamma", "vega", "theta", "rho"]}

    for S in spot_grid:
        scenario = BSInputs(S, base_inputs.strike, base_inputs.maturity, base_inputs.rate, base_inputs.volatility, base_inputs.dividend_yield)
        call_prices.append(call_price(scenario))
        put_prices.append(put_price(scenario))
        c_greeks = all_greeks(scenario, "call")
        p_greeks = all_greeks(scenario, "put")
        for k in call_greeks_grid:
            call_greeks_grid[k].append(c_greeks[k])
            put_greeks_grid[k].append(p_greeks[k])

    chart_paths = {
        "Price vs spot": plots.plot_price_vs_spot(spot_grid, call_prices, put_prices, output_dir),
        "Greeks vs spot": plots.plot_greeks_vs_spot(spot_grid, call_greeks_grid, put_greeks_grid, output_dir),
        "Binomial convergence": plots.plot_binomial_convergence(binom_convergence["steps"], binom_convergence["abs_errors"], output_dir),
        "Monte Carlo convergence": plots.plot_mc_convergence(mc_convergence["paths"], mc_convergence["abs_errors"], mc_convergence["std_errors"], output_dir),
        "Implied volatility smile": plots.plot_iv_smile(smile["strikes"], smile["true_vols"], smile["recovered_vols"], output_dir),
        "Hedging P&L distributions": plots.plot_hedging_pnl_distributions(pnl_by_key, output_dir),
    }

    results = {
        "base_scenario": base_scenario,
        "implied_vol_check": iv_check,
        "binomial_convergence": binom_convergence,
        "mc_convergence": mc_convergence,
        "iv_smile": smile,
        "hedging_summary": hedging_summary,
        "chart_paths": chart_paths,
    }

    report_path = report.build_report(results, output_dir)

    print("Pipeline complete.")
    print(f"Base call price: {base_scenario['call_price']:.4f}, put price: {base_scenario['put_price']:.4f}")
    print(f"Put-call parity diff: {base_scenario['parity_diff']:.2e}")
    print(f"Implied vol recovered: {iv_check['recovered_vol']:.6f} (input was {iv_check['input_vol']:.6f})")
    print(f"Report written to: {report_path}")
    print(f"Charts written to: {output_dir}/")


if __name__ == "__main__":
    main()
