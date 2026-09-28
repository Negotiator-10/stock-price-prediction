"""
TEST 3 of 3: does an edge hide in a particular market REGIME? (purged, Apple)
============================================================================
Even with no overall edge, maybe the model works in calm markets but not wild
ones, or in uptrends but not downtrends. We take the PURGED (leak-free) Apple
predictions and split them by regime, then measure edge over always-up in each.
Regimes use only past data at the prediction time (no look-ahead).
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

HORIZON, CORR_DROP, STEP, START, WINDOW = 5, 0.95, 5, 504, 63

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
close = df["Close(t)"]
daily = close.pct_change()
realized_vol = daily.rolling(20).std().values         # volatility at each day (past only)
ma200 = close.rolling(200).mean().values              # trend gauge (past only)
close_v = close.values

# ---- purged walk forward, recording regime info per prediction ----
recs = []
for t in range(START, len(df) - HORIZON, STEP):
    end = t - HORIZON
    lo = max(0, end - WINDOW)
    if end - lo < 10: continue
    sc = StandardScaler().fit(Xv[lo:end])
    m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xv[lo:end]), y[lo:end])
    pr = m.predict(sc.transform(Xv[t:t+1]))[0]
    recs.append({"correct": int((pr > 0) == (y[t] > 0)), "up": int(y[t] > 0),
                 "vol": realized_vol[t], "trend_up": int(close_v[t] > ma200[t])})
R = pd.DataFrame(recs)

def edge(sub):
    n = len(sub); corr = sub["correct"].sum()
    acc = corr / n * 100; base = sub["up"].mean() * 100
    p = binomtest(int(corr), n, sub["up"].mean(), alternative="greater").pvalue
    return {"n": n, "Dir%": round(acc,1), "always_up%": round(base,1),
            "edge_pts": round(acc-base,1), "sig?": "YES" if p < 0.05 else "no"}

vol_med = R["vol"].median()
groups = {
    "ALL": R,
    "low volatility":  R[R["vol"] <= vol_med],
    "high volatility": R[R["vol"] >  vol_med],
    "uptrend (>MA200)":   R[R["trend_up"] == 1],
    "downtrend (<MA200)": R[R["trend_up"] == 0],
}
print("=" * 76)
print("TEST 3: PURGED Apple model, edge over always-up BY REGIME")
print("=" * 76)
rows = [{"regime": k, **edge(v)} for k, v in groups.items() if len(v) > 20]
print(pd.DataFrame(rows).to_string(index=False))
print("\nNo positive, significant edge in any regime = the model has no skill even")
print("when conditions are split out. A negative edge just means no signal (noise).")
