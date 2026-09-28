"""
Robustness sweep: is the short-window edge real, or did we just get lucky at 63?
==============================================================================
For each training-window size, walk forward over the SAME test days, apply the
drift-adjusted significance test (edge over the always-up base rate). If the edge
holds across neighbouring windows, it is a genuine recency effect. If only 63
works, it was data-snooping on the window size.
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

HORIZON, CORR_DROP, STEP, START = 5, 0.95, 5, 504

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

def run(window):
    true_up, model_up = [], []
    for t in range(START, len(df) - HORIZON, STEP):
        lo = max(0, t - window)
        sc = StandardScaler().fit(Xv[lo:t])
        m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xv[lo:t]), y[lo:t])
        pr = m.predict(sc.transform(Xv[t:t+1]))[0]
        true_up.append(int(y[t] > 0)); model_up.append(int(pr > 0))
    true_up = np.array(true_up); model_up = np.array(model_up); n = len(true_up)
    correct = int((true_up == model_up).sum())
    acc = correct / n * 100
    base = true_up.mean() * 100
    p = binomtest(correct, n, true_up.mean(), alternative="greater").pvalue
    return acc, base, acc - base, p

print("=" * 72)
print("WINDOW-SIZE ROBUSTNESS SWEEP (drift-adjusted, same test days)")
print("=" * 72)
rows = []
for w in [21, 42, 63, 84, 126, 252, 504]:
    acc, base, edge, p = run(w)
    rows.append({"window_days": w, "Dir%": round(acc,1), "always_up%": round(base,1),
                 "edge_pts": round(edge,1), "p_vs_drift": round(p,4),
                 "sig?": "YES" if p < 0.05 else "no"})
print(pd.DataFrame(rows).to_string(index=False))
print("\nEdge holds across neighbours -> real recency effect.")
print("Only 63 works -> data-snooping on the window size.")
