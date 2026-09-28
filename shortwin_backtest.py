"""
TEST 1 of 3: does the short-window momentum edge survive transaction costs?
==========================================================================
Turn the short rolling-window signal into a real long/cash strategy, charge
realistic costs, and compare to buy-and-hold. Short windows trade more, so this
is the honest test of whether the directional edge becomes actual profit.

Strategy: step through non-overlapping 5-day blocks. Retrain Lasso on the last
WINDOW days, predict the 5-day return; long if predicted up, else cash. Pay a
cost every time the position changes.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

HORIZON, CORR_DROP, STEP, START = 5, 0.95, 5, 504
PERIODS_PER_YEAR = 252 / HORIZON

df = pd.read_csv("data/AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns: df = df.drop(columns=["Date_col"])
df["ret"] = df["Close(t)"].shift(-HORIZON) / df["Close(t)"] - 1
df = df.dropna(subset=["ret"]).reset_index(drop=True)
X = df.drop(columns=["Date", "Close_forcast", "ret"]).select_dtypes("number")
c = X.corr().abs(); up = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
X = X.drop(columns=[k for k in up.columns if (up[k] > CORR_DROP).any()])
Xv = X.values; y = df["ret"].values

def metrics(r):
    eq = np.cumprod(1 + r); total = eq[-1] - 1
    ann = (1 + total) ** (PERIODS_PER_YEAR / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR) if r.std() > 0 else 0.0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    return total, ann, sharpe, dd

def backtest(window, cost_bps):
    cost = cost_bps / 10000.0
    strat, bh, positions = [], [], []
    prev = 0
    for t in range(START, len(df) - HORIZON, STEP):
        lo = max(0, t - window)
        sc = StandardScaler().fit(Xv[lo:t])
        m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xv[lo:t]), y[lo:t])
        pos = 1 if m.predict(sc.transform(Xv[t:t+1]))[0] > 0 else 0
        r = y[t]
        strat.append(pos * r - abs(pos - prev) * cost)
        bh.append(r); positions.append(pos); prev = pos
    strat = np.array(strat); bh = np.array(bh); positions = np.array(positions)
    n_trades = int(np.sum(np.abs(np.diff(np.concatenate([[0], positions])))))
    return strat, bh, n_trades, positions.mean()

print("=" * 78)
print("TEST 1: SHORT-WINDOW STRATEGY vs BUY-AND-HOLD, after costs")
print("=" * 78)
# buy-and-hold reference (from any run, cost-free bh)
_, bh0, _, _ = backtest(63, 0)
bt, ba, bs, bdd = metrics(bh0)
print(f"\nBUY-AND-HOLD:  total {bt*100:.0f}%   annual {ba*100:.1f}%   Sharpe {bs:.2f}   maxDD {bdd*100:.0f}%\n")

for window in [21, 63]:
    print(f"--- {window}-day window ---")
    rows = []
    for c_bps in [0, 5, 10, 20]:
        strat, _, n_trades, invested = backtest(window, c_bps)
        t, a, s, dd = metrics(strat)
        rows.append({"cost_bps": c_bps, "total%": round(t*100), "annual%": round(a*100,1),
                     "Sharpe": round(s,2), "maxDD%": round(dd*100), "trades": n_trades,
                     "%invested": round(invested*100)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
print("=" * 78)
print("Read: compare the strategy's after-cost annual return and Sharpe to buy-and-hold.")
print("More trades (short window) = more cost drag. Does the edge still beat holding?")
