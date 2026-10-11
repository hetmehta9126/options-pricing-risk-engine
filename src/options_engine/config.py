"""
Central place for the assumptions and parameters used across the project.

Everything here is a modeling assumption or a default input, not a market
observation. Change the numbers below to re-run the whole pipeline with
different assumptions.
"""

# Risk-free rate assumption.
# Set to the RBI 91-day T-bill cut-off yield from the 7 Oct 2026 auction
# (5.5747%, rounded to 5.57%). This is used as a constant, continuously
# compounded risk-free rate everywhere in this project. It is NOT pulled
# live from any source, it is just a fixed number we picked once.
RISK_FREE_RATE = 0.0557

# Default market-like inputs used to build the example pricing scenarios.
# All of these are made up for demonstration purposes, not real quotes.
DEFAULT_SPOT = 100.0
DEFAULT_STRIKE = 100.0
DEFAULT_MATURITY = 1.0          # years
DEFAULT_VOLATILITY = 0.20       # 20% annualized
DEFAULT_DIVIDEND_YIELD = 0.0    # continuous dividend yield

# Grid used for "price / Greeks vs spot" charts.
SPOT_GRID_MIN = 50.0
SPOT_GRID_MAX = 150.0
SPOT_GRID_POINTS = 101

# Binomial tree convergence study: number of steps to try.
BINOMIAL_STEPS_GRID = [5, 10, 20, 50, 100, 200, 400, 800]

# Monte Carlo convergence study: number of paths to try.
MC_PATHS_GRID = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

# Seed used everywhere a random draw happens, so every run is reproducible.
RANDOM_SEED = 42

# Delta hedging simulation settings.
HEDGE_NUM_PATHS = 2_000
HEDGE_NUM_STEPS_PER_YEAR = 252  # trading days, used as the finest grid
HEDGE_REBALANCE_FREQUENCIES = {
    "daily": 1,            # rebalance every trading day
    "weekly": 5,            # every 5 trading days
    "monthly": 21,           # every 21 trading days
}
HEDGE_TRANSACTION_COST_RATES = [0.0, 0.0005, 0.002]  # proportional, per trade

# Implied volatility solver settings.
IV_INITIAL_GUESS = 0.20
IV_MAX_ITER_NEWTON = 50
IV_MAX_ITER_BISECTION = 200
IV_TOLERANCE = 1e-8
IV_BOUNDS = (1e-4, 5.0)  # vol search bounds for the bisection fallback

# Synthetic volatility skew model used to build a fake "smile" for the IV
# recovery demo. This is NOT fit to any real market, it's a toy shape:
#   sigma(K) = base_vol + skew_slope * (K/S - 1) + smile_curv * (K/S - 1)^2
SMILE_BASE_VOL = 0.20
SMILE_SKEW_SLOPE = -0.10
SMILE_CURVATURE = 0.15
SMILE_STRIKE_GRID_MIN = 70.0
SMILE_STRIKE_GRID_MAX = 130.0
SMILE_STRIKE_GRID_POINTS = 25

OUTPUT_DIR = "outputs"
