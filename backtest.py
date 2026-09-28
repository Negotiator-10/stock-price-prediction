"""
Transaction-cost backtest: does the Option B signal actually make money?
========================================================================
Predicting a return is not the same as profiting from it. This script turns the
Option B signal (Lasso predicting the 5-day return) into a simple, honest trading
strategy, charges realistic transaction costs, and compares it to just buying and
holding Apple.

Strategy (kept deliberately simple and non-overlapping):
  - Step through the out-of-sample test period in non-overlapping 5-day blocks.
  - At the start of each block, look at the predicted 5-day return.
      predicted up   -> hold Apple for the next 5 days (long)
      predicted down -> hold cash for the next 5 days (flat, earns 0)
  - Pay a transaction cost every time the position changes (enter or exit).

Benchmark: buy and hold Apple for the whole test period (one entry cost).

We report total return, annualised return, Sharpe ratio, max drawdown and the
number of trades, at several cost levels, so you can see how fast any edge is
eaten by costs.

Run:
    .venv\\Scripts\\python.exe backtest.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

DATA_PATH = r"data/AAPL.csv"
OUT_DIR   = r"outputs"
HORIZON, CORR_DROP, TRAIN_FRAC = 5, 0.95, 0.80
PERIODS_PER_YEAR = 252 / HORIZON          # about 50 five-day blocks per year
os.makedirs(OUT_DIR, exist_ok=True)

# ---- data + return target (Option B) ----
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns:
    df = df.drop(columns=["Date_col"])
df["target_return"] = df["Close(t)"].shift(-HORIZON) / df["Close(t)"] - 1
df = df.dropna(subset=["target_return"]).reset_index(drop=True)

leak = ["Date", "Close_forcast", "target_return"]
X = df.drop(columns=[c for c in leak if c in df.columns]).select_dtypes(include=[np.number])
cmat = X.corr().abs()
upper = cmat.where(np.triu(np.ones(cmat.shape), k=1).astype(bool))
X = X.drop(columns=[c for c in upper.columns if (upper[c] > CORR_DROP).any()])
y = df["target_return"]

# ---- train on the first 80 percent, predict returns on the last 20 percent ----
split = int(len(df) * TRAIN_FRAC)
scaler = StandardScaler().fit(X.iloc[:split])
Xtr, Xte = scaler.transform(X.iloc[:split]), scaler.transform(X.iloc[split:])
lasso = GridSearchCV(Lasso(max_iter=5000), {"alpha": [0.0005, 0.001, 0.01, 0.1]},
                     cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error").fit(Xtr, y.iloc[:split])
pred_ret = lasso.predict(Xte)                       # predicted 5-day return, per test day

actual_ret = df["target_return"].iloc[split:].values   # actual 5-day return, per test day
dates      = df["Date"].iloc[split:].values

# ---- run the strategy over non-overlapping 5-day blocks ----
def run_strategy(cost_bps, threshold=0.0):
    """Go long when the predicted 5-day return exceeds `threshold`, else hold cash.
    cost_bps is charged per position change (one-way). Returns per-block series."""
    cost = cost_bps / 10000.0
    steps = range(0, len(pred_ret) - HORIZON, HORIZON)   # non-overlapping blocks
    strat, bh, block_dates, invested = [], [], [], []
    prev_pos = 0
    for i in steps:
        pos = 1 if pred_ret[i] > threshold else 0        # long only if signal clears the bar
        r = actual_ret[i]                                # the actual 5-day return of that block
        trade = abs(pos - prev_pos)                      # 1 if we changed position, else 0
        strat.append(pos * r - trade * cost)             # strategy return, net of cost
        bh.append(r)                                     # buy-and-hold is always long
        block_dates.append(dates[i]); invested.append(pos)
        prev_pos = pos
    bh = np.array(bh); strat = np.array(strat)
    bh[0] -= cost                                        # buy-and-hold pays one entry cost
    return np.array(block_dates), strat, bh, np.array(invested)

def metrics(returns):
    eq = np.cumprod(1 + returns)                         # equity curve
    total = eq[-1] - 1
    ann = (1 + total) ** (PERIODS_PER_YEAR / len(returns)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR) if returns.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = ((eq - peak) / peak).min()
    return total, ann, sharpe, max_dd

print("=" * 74)
print("TRANSACTION-COST BACKTEST  (test period", pd.Timestamp(dates[0]).date(),
      "to", pd.Timestamp(dates[-1]).date(), ")")
print("=" * 74)

# buy-and-hold reference
_, _, bh0, _ = run_strategy(0)
bh_t, bh_a, bh_s, bh_dd = metrics(bh0)

# ---- Part 1: costs, on the plain "long when predicted up" rule (threshold 0) ----
rows = []
for c in [0, 5, 10, 20]:
    _, strat, _, inv = run_strategy(c, threshold=0.0)
    n_trades = int(np.sum(np.abs(np.diff(np.concatenate([[0], inv])))))
    t, a, s, dd = metrics(strat)
    rows.append({"cost_bps/trade": c, "total%": round(t*100,1), "ann%": round(a*100,1),
                 "Sharpe": round(s,2), "maxDD%": round(dd*100,1), "trades": n_trades,
                 "%time_invested": round(inv.mean()*100,0)})
bt = pd.DataFrame(rows)
print("\nPART 1  Rule: long when predicted return > 0, else cash.")
print(bt.to_string(index=False))
print("This rule stays invested ~100%% of the time and trades once: it is just")
print("buy-and-hold in disguise. The signal never actually times the market.")

# ---- Part 2: be more selective (raise the bar). Does picking spots beat buy-and-hold? ----
rows2 = []
for thr in [0.0, 0.005, 0.01, 0.02, 0.03]:
    _, strat, _, inv = run_strategy(10, threshold=thr)   # realistic 10 bps
    n_trades = int(np.sum(np.abs(np.diff(np.concatenate([[0], inv])))))
    t, a, s, dd = metrics(strat)
    rows2.append({"long_if_pred>": f"{thr*100:.1f}%", "total%": round(t*100,1),
                  "ann%": round(a*100,1), "Sharpe": round(s,2), "maxDD%": round(dd*100,1),
                  "trades": n_trades, "%time_invested": round(inv.mean()*100,0)})
bt2 = pd.DataFrame(rows2)
print("\nPART 2  Try to be selective (10 bps cost). Higher bar = more time in cash.")
print(bt2.to_string(index=False))

print("\nBUY-AND-HOLD benchmark (always long):")
print("  total%%: %.1f   annual%%: %.1f   Sharpe: %.2f   maxDD%%: %.1f"
      % (bh_t*100, bh_a*100, bh_s, bh_dd*100))
print("=" * 74)

bt.to_csv(os.path.join(OUT_DIR, "backtest_costs.csv"), index=False)
bt2.to_csv(os.path.join(OUT_DIR, "backtest_thresholds.csv"), index=False)

# ---- chart 1: equity curve, strategy vs buy-and-hold (they overlap) ----
bdates, strat10, bh10, _ = run_strategy(10, threshold=0.0)
plt.figure(figsize=(11, 5))
plt.plot(bdates, np.cumprod(1 + bh10), label="Buy and hold Apple", linewidth=3, color="steelblue")
plt.plot(bdates, np.cumprod(1 + strat10), label="Strategy (Option B signal, 10 bps)",
         linewidth=1.5, color="orange", linestyle="--")
plt.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
plt.ylabel("growth of 1 dollar")
plt.title("Strategy vs buy-and-hold: the two lines sit on top of each other")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "backtest_equity.png"))
plt.close()

# ---- chart 2: predicted vs actual returns over time (the WHY) ----
# The predicted line is flat: the model collapsed to a constant (~0.63%), ignoring every feature.
plt.figure(figsize=(11, 5))
plt.plot(dates, actual_ret, color="steelblue", linewidth=0.8, alpha=0.7, label="actual 5-day return")
plt.plot(dates, pred_ret, color="red", linewidth=2.5,
         label=f"model prediction (flat at {pred_ret.mean()*100:.2f}%)")
plt.axhline(0, color="grey", linestyle=":", linewidth=0.8)
plt.ylabel("5-day return")
plt.title("The model collapsed to a constant: it predicts the same number every day, ignoring all features")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "backtest_signal_flat.png"))
plt.close()
print("saved backtest_costs.csv, backtest_thresholds.csv, backtest_equity.png, backtest_signal_flat.png")
