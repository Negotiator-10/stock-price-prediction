"""
DIG #3: direction as a proper CLASSIFIER (purged), with AUC and threshold view
=============================================================================
Cleaner framing than regression-then-sign: a logistic classifier predicts the
PROBABILITY of an up move. ROC-AUC measures whether that probability ranks real
up-days above down-days better than chance (0.5), independent of any threshold.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

H, CORR_DROP, WINDOW, START = 5, 0.95, 500, 520

df = pd.read_csv("data/AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns: df = df.drop(columns=["Date_col"])
df["ret"] = df["Close(t)"].shift(-H) / df["Close(t)"] - 1
df = df.dropna(subset=["ret"]).reset_index(drop=True)
df["up"] = (df["ret"] > 0).astype(int)
X = df.drop(columns=["Date", "Close_forcast", "ret", "up"]).select_dtypes("number")
c = X.corr().abs(); u = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
X = X.drop(columns=[k for k in u.columns if (u[k] > CORR_DROP).any()])
Xv = X.values; y = df["up"].values

probs, actual = [], []
for t in range(START, len(df) - H, H):              # non-overlapping
    end = t - H                                     # purge
    lo = max(0, end - WINDOW)
    if end - lo < 40 or len(np.unique(y[lo:end])) < 2: continue
    sc = StandardScaler().fit(Xv[lo:end])
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xv[lo:end]), y[lo:end])
    probs.append(m.predict_proba(sc.transform(Xv[t:t+1]))[0, 1])
    actual.append(y[t])
probs = np.array(probs); actual = np.array(actual)

auc = roc_auc_score(actual, probs)
# bootstrap a rough 95% CI for AUC
rng = np.random.default_rng(0); boots = []
for _ in range(2000):
    idx = rng.integers(0, len(actual), len(actual))
    if len(np.unique(actual[idx])) < 2: continue
    boots.append(roc_auc_score(actual[idx], probs[idx]))
lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])

print("=" * 64)
print("DIG #3: directional CLASSIFIER (purged),", len(actual), "predictions")
print("=" * 64)
print(f"ROC-AUC                 : {auc:.3f}")
print(f"95% bootstrap CI        : [{lo_ci:.3f}, {hi_ci:.3f}]")
print(f"chance level            : 0.500")
print(f"always-up base rate     : {actual.mean()*100:.1f}%")
verdict = "BEATS chance (CI above 0.5)" if lo_ci > 0.5 else "NO better than chance (CI includes 0.5)"
print(f"\nVerdict: {verdict}")
