"""
Does the 63-day rolling window's 67.7% direction survive the honest tests?
=========================================================================
Same drift + significance gauntlet we used on the original signal, applied to
the short-window model:
  1. DRIFT: how often did the price actually rise on these days? (always-up bar)
  2. EDGE: model accuracy minus the always-up base rate.
  3. SIGNIFICANCE: binomial test vs a coin flip AND vs always-up.
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

true_up, model_up = [], []
for t in range(START, len(df) - HORIZON, STEP):
    lo = max(0, t - WINDOW)
    sc = StandardScaler().fit(Xv[lo:t])
    m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xv[lo:t]), y[lo:t])
    pr = m.predict(sc.transform(Xv[t:t+1]))[0]
    true_up.append(int(y[t] > 0))            # did the price actually rise over 5 days?
    model_up.append(int(pr > 0))             # did the model predict up?

true_up = np.array(true_up); model_up = np.array(model_up); n = len(true_up)
correct = int((true_up == model_up).sum())
model_acc = correct / n * 100
base_up = true_up.mean() * 100                # always-up accuracy (the drift bar)

p_coin = binomtest(correct, n, 0.5, alternative="greater").pvalue
p_drift = binomtest(correct, n, true_up.mean(), alternative="greater").pvalue
verdict = lambda p: "SIGNIFICANT (p<0.05)" if p < 0.05 else "NOT significant (could be luck)"

print("=" * 66)
print("63-DAY ROLLING WINDOW: is the direction edge real? (", n, "predictions )")
print("=" * 66)
print("Model directional accuracy :", round(model_acc, 1), "%")
print("Always-up base rate (drift):", round(base_up, 1), "%  <- the honest bar")
print("Edge OVER always-up        :", round(model_acc - base_up, 1), "percentage points")
print()
print("Better than a COIN FLIP?  p =", round(p_coin, 4), "->", verdict(p_coin))
print("Better than ALWAYS-UP?    p =", round(p_drift, 4), "->", verdict(p_drift))
print("=" * 66)
