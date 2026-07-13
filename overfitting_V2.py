"""
Overfitting trap demo - moving average crossover on an EM Asia ETF.

Idea: optimise a simple trading strategy naively on historical data,
then test the "best" parameters on unseen data to see how much of the
performance was real vs just noise fitting.

Extended with walk-forward validation (multiple train/test windows),
transaction costs, and a simple correction attempt (averaging the top-N
parameter combos instead of keeping only the best one).

Usage:
    pip install -r requirements.txt
    python overfitting_demo_v2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

USE_REAL_DATA = True
TICKER = "AAXJ"  # proxy for Emerging Markets Asia, liquid and on Yahoo Finance

np.random.seed(1)


def generate_price_series(n_days=2600, s0=100):
    # fallback series with a few regime changes, only used if USE_REAL_DATA is False
    n_regimes = 5
    regime_len = n_days // n_regimes
    all_returns = []
    regime_params = [
        (0.0003, 0.014, 0.15),
        (0.0005, 0.010, 0.0),
        (-0.0002, 0.018, 0.10),
        (0.0004, 0.011, 0.0),
        (0.0002, 0.015, 0.12),
    ]
    for mu, sigma, autocorr_strength in regime_params:
        r = np.random.normal(mu, sigma, regime_len)
        if autocorr_strength > 0:
            noise = np.random.normal(0, sigma * 0.5, regime_len)
            r += autocorr_strength * np.roll(noise, 1)
        all_returns.append(r)
    returns = np.concatenate(all_returns)
    prices = s0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2016-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=dates)


if USE_REAL_DATA:
    import yfinance as yf
    data = yf.download(TICKER, start="2016-01-01", end="2025-01-01", progress=False)
    prices = data["Close"].dropna()
    if isinstance(prices, pd.DataFrame):
        # recent yfinance versions sometimes return a MultiIndex column
        # even for a single ticker - squeeze it down to a plain Series
        prices = prices.iloc[:, 0]
    prices.index = pd.to_datetime(prices.index)
else:
    prices = generate_price_series()

print(f"Total price history: {len(prices)} trading days "
      f"({prices.index[0].date()} -> {prices.index[-1].date()})")


# --- strategy ---

TRANSACTION_COST = 0.0010  # 10 bps per position change, rough broker estimate

def backtest_ma_crossover(price_series, short_window, long_window, cost=0.0):
    df = pd.DataFrame({"price": price_series})
    df["ma_short"] = df["price"].rolling(short_window).mean()
    df["ma_long"] = df["price"].rolling(long_window).mean()
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)
    df["daily_return"] = df["price"].pct_change()

    # use yesterday's signal so we don't trade on info we wouldn't have had yet
    df["position"] = df["signal"].shift(1)
    df["strategy_return"] = df["position"] * df["daily_return"]

    # charge a cost every time the position changes
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_return_net"] = df["strategy_return"] - df["trade"] * cost

    df = df.dropna()
    cumulative_return = (1 + df["strategy_return_net"]).prod() - 1
    n_trades = int(df["trade"].sum())
    return cumulative_return, n_trades


# --- walk-forward windows ---

TRAIN_LEN = 500   # ~2 years
TEST_LEN = 250    # ~1 year
STEP = TEST_LEN   # test windows don't overlap

short_windows = range(3, 30, 3)
long_windows = range(20, 150, 10)

windows = []
start = 0
while start + TRAIN_LEN + TEST_LEN <= len(prices):
    train = prices.iloc[start: start + TRAIN_LEN]
    test = prices.iloc[start + TRAIN_LEN: start + TRAIN_LEN + TEST_LEN]
    windows.append((train, test))
    start += STEP

print(f"Number of walk-forward windows: {len(windows)}")


# --- run naive optimisation + ensemble correction on each window ---
# Two scenarios: with transaction costs, and without (matches my own PEA,
# where this ETF trades commission-free)

TOP_N = 5

def run_walk_forward(windows, cost):
    records = []
    for w_idx, (train, test) in enumerate(windows):
        grid_results = []
        for sw in short_windows:
            for lw in long_windows:
                if sw >= lw:
                    continue
                ret, _ = backtest_ma_crossover(train, sw, lw, cost=cost)
                grid_results.append((sw, lw, ret))
        grid_df = pd.DataFrame(grid_results, columns=["sw", "lw", "train_return"])
        grid_df = grid_df.sort_values("train_return", ascending=False).reset_index(drop=True)

        # naive: just take the single best combo found
        best = grid_df.iloc[0]
        test_ret_naive, _ = backtest_ma_crossover(test, int(best["sw"]), int(best["lw"]), cost=cost)
        gen_ratio_naive = test_ret_naive / best["train_return"] if best["train_return"] != 0 else np.nan

        # correction attempt: average the parameters of the top N combos instead
        top_n = grid_df.head(TOP_N)
        corrected_sw = int(round(top_n["sw"].mean()))
        corrected_lw = int(round(top_n["lw"].mean()))
        if corrected_sw >= corrected_lw:
            corrected_lw = corrected_sw + 5

        train_ret_ensemble, _ = backtest_ma_crossover(train, corrected_sw, corrected_lw, cost=cost)
        test_ret_ensemble, _ = backtest_ma_crossover(test, corrected_sw, corrected_lw, cost=cost)
        gen_ratio_ensemble = test_ret_ensemble / train_ret_ensemble if train_ret_ensemble != 0 else np.nan

        records.append({
            "window": w_idx + 1,
            "test_period": f"{test.index[0].date()} to {test.index[-1].date()}",
            "naive_train_return": best["train_return"],
            "naive_test_return": test_ret_naive,
            "naive_gen_ratio": gen_ratio_naive,
            "ensemble_train_return": train_ret_ensemble,
            "ensemble_test_return": test_ret_ensemble,
            "ensemble_gen_ratio": gen_ratio_ensemble,
        })
    return pd.DataFrame(records)


print("\nScenario A - with transaction costs (10 bps)")
results_with_cost = run_walk_forward(windows, cost=TRANSACTION_COST)

print("Scenario B - without transaction costs (my real PEA setup)")
results_no_cost = run_walk_forward(windows, cost=0.0)

results_df = results_with_cost  # used below for the main plot

pd.set_option("display.width", 120)
print("\nResults with transaction costs:")
print(results_with_cost[["window", "test_period", "naive_gen_ratio", "ensemble_gen_ratio"]].to_string(index=False))

print("\nResults without transaction costs:")
print(results_no_cost[["window", "test_period", "naive_gen_ratio", "ensemble_gen_ratio"]].to_string(index=False))

print("\n--- summary, with costs ---")
print(f"Naive mean generalisation ratio:    {results_with_cost['naive_gen_ratio'].mean():.3f} "
      f"(std {results_with_cost['naive_gen_ratio'].std():.3f})")
print(f"Ensemble mean generalisation ratio: {results_with_cost['ensemble_gen_ratio'].mean():.3f} "
      f"(std {results_with_cost['ensemble_gen_ratio'].std():.3f})")
n_naive_negative = (results_with_cost["naive_test_return"] < 0).sum()
n_ensemble_negative = (results_with_cost["ensemble_test_return"] < 0).sum()
print(f"Negative out-of-sample windows: naive {n_naive_negative}/{len(results_with_cost)}, "
      f"ensemble {n_ensemble_negative}/{len(results_with_cost)}")

print("\n--- summary, without costs ---")
print(f"Naive mean generalisation ratio:    {results_no_cost['naive_gen_ratio'].mean():.3f} "
      f"(std {results_no_cost['naive_gen_ratio'].std():.3f})")
print(f"Ensemble mean generalisation ratio: {results_no_cost['ensemble_gen_ratio'].mean():.3f} "
      f"(std {results_no_cost['ensemble_gen_ratio'].std():.3f})")
n_naive_negative_nc = (results_no_cost["naive_test_return"] < 0).sum()
n_ensemble_negative_nc = (results_no_cost["ensemble_test_return"] < 0).sum()
print(f"Negative out-of-sample windows: naive {n_naive_negative_nc}/{len(results_no_cost)}, "
      f"ensemble {n_ensemble_negative_nc}/{len(results_no_cost)}")

delta_naive = results_with_cost['naive_gen_ratio'].mean() - results_no_cost['naive_gen_ratio'].mean()
print(f"\nCost impact on naive generalisation ratio: {delta_naive:+.3f}")
print("(near zero means the overfitting problem isn't really about trading costs)")


# --- plots ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

x = results_df["window"]
width = 0.35
axes[0].bar(x - width/2, results_df["naive_gen_ratio"], width, label="Naive (best single combo)", color="#E74C3C")
axes[0].bar(x + width/2, results_df["ensemble_gen_ratio"], width, label=f"Ensemble (top {TOP_N})", color="#27AE60")
axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Perfect generalisation")
axes[0].axhline(0.0, color="grey", linewidth=0.8)
axes[0].set_xlabel("Walk-forward window")
axes[0].set_ylabel("Generalisation ratio")
axes[0].set_title("Generalisation ratio per window (with transaction costs)")
axes[0].legend()
axes[0].set_xticks(x)

methods = ["Naive\nmean ratio", "Ensemble\nmean ratio"]
values = [results_df["naive_gen_ratio"].mean(), results_df["ensemble_gen_ratio"].mean()]
colors = ["#E74C3C", "#27AE60"]
axes[1].bar(methods, values, color=colors)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
axes[1].axhline(0.0, color="grey", linewidth=0.8)
axes[1].set_ylabel("Mean generalisation ratio")
axes[1].set_title("Naive selection vs ensemble correction")
for i, v in enumerate(values):
    axes[1].text(i, v + 0.03 * np.sign(v if v != 0 else 1), f"{v:.2f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("overfitting_demo_v2.png", dpi=130)
plt.close()
print("\nSaved overfitting_demo_v2.png")

fig2, ax = plt.subplots(figsize=(8, 5.5))
scenarios = ["With costs\n(10 bps)", "Without costs\n(my real PEA setup)"]
naive_means = [results_with_cost["naive_gen_ratio"].mean(), results_no_cost["naive_gen_ratio"].mean()]
ensemble_means = [results_with_cost["ensemble_gen_ratio"].mean(), results_no_cost["ensemble_gen_ratio"].mean()]

x = np.arange(len(scenarios))
width = 0.35
ax.bar(x - width/2, naive_means, width, label="Naive (best single combo)", color="#E74C3C")
ax.bar(x + width/2, ensemble_means, width, label="Ensemble (top 5)", color="#27AE60")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Perfect generalisation")
ax.axhline(0.0, color="grey", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Mean generalisation ratio")
ax.set_title("Does removing transaction costs fix the overfitting problem?")
ax.legend()
for i, v in enumerate(naive_means):
    ax.text(i - width/2, v + 0.02*np.sign(v if v != 0 else 1), f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
for i, v in enumerate(ensemble_means):
    ax.text(i + width/2, v + 0.02*np.sign(v if v != 0 else 1), f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("cost_impact_comparison.png", dpi=130)
plt.close()
print("Saved cost_impact_comparison.png")
