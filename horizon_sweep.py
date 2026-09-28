"""
DIG #2: horizon sweep -- does ANY forecast horizon show real directional signal?
===============================================================================
We only tried 1 and 5 days. Try 1, 5, 10, 20, 60. For each, run a PURGED
walk-forward directional test and measure the edge over the always-up baseline
(the honest bar). Longer horizons sometimes carry mean-reversion or value effects.
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

CORR_DROP, WINDOW, START = 0.95, 500, 520

base = pd.read_csv("data/AAPL.csv")
base["Date"] = pd.to_datetime(base["Date"], format="%d-%m-%Y")
base = base.sort_values("Date").reset_index(drop=True)
if "Date_col" in base.columns: base = base.drop(columns=["Date_col"])

def run_horizon(H):
    df = base.copy()
    df["ret"] = df["Close(t)"].shift(-H) / df["Close(t)"] - 1
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    X = df.drop(columns=["Date", "Close_forcast", "ret"]).select_dtypes("number")
    c = X.corr().abs(); up = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
    X = X.drop(columns=[k for k in up.columns if (up[k] > CORR_DROP).any()])
    Xv = X.values; y = df["ret"].values
    tu, mu = [], []
    for t in range(START, len(df) - H, H):          # step by H = non-overlapping = independent
        end = t - H                                 # PURGE: labels known before t
        lo = max(0, end - WINDOW)
        if end - lo < 30: continue
        sc = StandardScaler().fit(Xv[lo:end])
        m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xv[lo:end]), y[lo:end])
        pr = m.predict(sc.transform(Xv[t:t+1]))[0]
        tu.append(int(y[t] > 0)); mu.append(int(pr > 0))
    tu = np.array(tu); mu = np.array(mu); n = len(tu)
    corr = int((tu == mu).sum()); acc = corr/n*100; bar = tu.mean()*100
    p = binomtest(corr, n, tu.mean(), alternative="greater").pvalue
    return {"horizon": H, "n": n, "Dir%": round(acc,1), "always_up%": round(bar,1),
            "edge_pts": round(acc-bar,1), "sig?": "YES" if p < 0.05 else "no"}

print("=" * 68)
print("DIG #2: horizon sweep (PURGED, edge over always-up)")
print("=" * 68)
rows = [run_horizon(H) for H in [1, 5, 10, 20, 60]]
print(pd.DataFrame(rows).to_string(index=False))
print("\nAny positive & significant edge = a horizon worth exploring. All near/below 0")
print("= direction is unpredictable at every horizon too.")
