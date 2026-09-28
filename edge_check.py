"""
Is the directional edge REAL? Two honest confirmation tests
===========================================================
The walk-forward test showed Lasso (predicting the 5-day return) called direction
right about 58% of the time. Two questions remain:

  1. DRIFT CHECK: Apple trended up for years, so "up" days are common anyway.
     If you just always guessed "up", how often would you be right? The model's
     real skill is only the amount by which it BEATS that always-up base rate.

  2. SIGNIFICANCE: is the model's hit rate reliably different from a coin flip
     (and from always-up), or could it be random luck? A binomial test gives a
     p-value.

Run:
    .venv\\Scripts\\python.exe edge_check.py
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

DATA_PATH = r"data/AAPL.csv"
HORIZON, CORR_DROP, N_FOLDS = 5, 0.95, 6

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

# ---- collect the model's directional calls across all walk-forward test days ----
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
true_up, model_up = [], []           # 1 = up, 0 = down, for each test day
for tr, te in tscv.split(X):
    scaler = StandardScaler().fit(X.iloc[tr])
    Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
    lasso = GridSearchCV(Lasso(max_iter=5000), {"alpha": [0.0005, 0.001, 0.01, 0.1]},
                         cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error").fit(Xtr, y.iloc[tr])
    pred_ret = lasso.predict(Xte)
    true_up.extend((y.iloc[te].values > 0).astype(int))     # did the price actually go up?
    model_up.extend((pred_ret > 0).astype(int))             # did the model say up?

true_up = np.array(true_up)
model_up = np.array(model_up)
n = len(true_up)

# ---- the three hit rates ----
model_correct = int((true_up == model_up).sum())            # model right (up or down)
model_acc = model_correct / n * 100

base_up_rate = true_up.mean() * 100                         # how often price actually went up
# "always up" predictor is right exactly on the days that were up:
alwaysup_correct = int(true_up.sum())
alwaysup_acc = alwaysup_correct / n * 100

print("=" * 64)
print("EDGE CONFIRMATION  (", n, "out-of-sample predictions )")
print("=" * 64)
print("Model directional accuracy      :", round(model_acc, 1), "%")
print("Always-up base rate (drift)     :", round(alwaysup_acc, 1), "%  <- the honest bar")
print("Coin flip                       : 50.0 %")
print("Model edge OVER always-up       :", round(model_acc - alwaysup_acc, 1), "percentage points")
print()

# ---- significance tests (one-sided: is the model BETTER than the bar?) ----
p_vs_coin = binomtest(model_correct, n, 0.5, alternative="greater").pvalue
p_vs_drift = binomtest(model_correct, n, true_up.mean(), alternative="greater").pvalue

def verdict(p):
    return "SIGNIFICANT (p < 0.05)" if p < 0.05 else "NOT significant (could be luck)"

print("Is the model better than a COIN FLIP?")
print("  p-value:", round(p_vs_coin, 4), "->", verdict(p_vs_coin))
print()
print("Is the model better than ALWAYS-UP (the real test)?")
print("  p-value:", round(p_vs_drift, 4), "->", verdict(p_vs_drift))
print("=" * 64)
