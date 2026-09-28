"""
The corrected test: robustness sweep WITH PURGING (no look-ahead)
================================================================
Earlier rolling code trained on y[lo:t], but y[i] = return from i to i+HORIZON is
only known at i+HORIZON. So the last HORIZON training rows peeked into the future
(look-ahead leakage), worst for short windows (a bigger fraction contaminated).

Fix (purging): at prediction time t, only train on rows whose label is already
known, i.e. rows i with i + HORIZON <= t. So train on [.. : t - HORIZON].

If the momentum edge was real, it should survive. If it was leakage, it vanishes.
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

def run(window, purge):
    true_up, model_up = [], []
    for t in range(START, len(df) - HORIZON, STEP):
        end = t - HORIZON if purge else t          # purge the look-ahead rows or not
        lo = max(0, end - window)
        if end - lo < 10:                          # need a few clean rows
            continue
        sc = StandardScaler().fit(Xv[lo:end])
        m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xv[lo:end]), y[lo:end])
        pr = m.predict(sc.transform(Xv[t:t+1]))[0]
        true_up.append(int(y[t] > 0)); model_up.append(int(pr > 0))
    true_up = np.array(true_up); model_up = np.array(model_up); n = len(true_up)
    correct = int((true_up == model_up).sum())
    acc = correct / n * 100; base = true_up.mean() * 100
    p = binomtest(correct, n, true_up.mean(), alternative="greater").pvalue
    return acc, base, acc - base, p

print("=" * 82)
print("PURGED ROBUSTNESS SWEEP: leaky vs corrected (edge = points over always-up)")
print("=" * 82)
rows = []
for w in [21, 42, 63, 84, 126, 252]:
    la, ba, ea, pa = run(w, purge=False)       # old, leaky
    lc, bc, ec, pc = run(w, purge=True)        # corrected, purged
    rows.append({"window": w,
                 "leaky_edge": round(ea,1), "leaky_sig": "YES" if pa<0.05 else "no",
                 "PURGED_edge": round(ec,1), "PURGED_sig": "YES" if pc<0.05 else "no"})
print(pd.DataFrame(rows).to_string(index=False))
print("\nIf PURGED_edge collapses toward 0 and PURGED_sig = no, the 'momentum edge'")
print("was look-ahead leakage, not a real signal.")
