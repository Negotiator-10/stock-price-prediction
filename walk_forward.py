"""
Walk-forward test of the Option B edge
======================================
The single 80/20 split said Lasso (predicting the 5-day RETURN) beat the naive
baseline with 63.7% directional accuracy. But that is ONE test period, so it
could be luck. This script checks whether the edge holds up across MANY periods.

How walk-forward works (expanding window):
  Fold 1: train on the oldest chunk, test on the next chunk.
  Fold 2: extend training to include that chunk, test on the following one.
  ... and so on, marching forward through history.
Each fold is a fresh, honest out-of-sample test. If the edge is real, Lasso
should beat naive in most folds. If it only wins once, it was a fluke.

We score on the PRICE scale (convert the predicted return back to a price) so
every fold is compared to the same naive baseline, exactly like the main script.

Run:
    .venv\\Scripts\\python.exe walk_forward.py
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
HORIZON   = 5
CORR_DROP = 0.95
N_FOLDS   = 6      # how many walk-forward test chunks
os.makedirs(OUT_DIR, exist_ok=True)


# ---- load and build the return target (Option B) ----
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns:
    df = df.drop(columns=["Date_col"])

df["target_price"]  = df["Close(t)"].shift(-HORIZON)                  # price in 5 days (for scoring)
df["target_return"] = df["Close(t)"].shift(-HORIZON) / df["Close(t)"] - 1  # 5-day return (what we learn)
df = df.dropna(subset=["target_price"]).reset_index(drop=True)


# ---- features: same recipe as the main script ----
leak_or_meta = ["Date", "Close_forcast", "target_price", "target_return"]
X = df.drop(columns=[c for c in leak_or_meta if c in df.columns]).select_dtypes(include=[np.number])
cmat = X.corr().abs()
upper = cmat.where(np.triu(np.ones(cmat.shape), k=1).astype(bool))
X = X.drop(columns=[c for c in upper.columns if (upper[c] > CORR_DROP).any()])

y_return = df["target_return"]                 # learning target
price5   = df["target_price"].values           # true future price (for scoring)
close_t  = df["Close(t)"].values               # today's close (naive + direction + convert back)


def score(y_true_price, y_pred_price, today_close):
    """RMSE in dollars and directional accuracy in percent."""
    rmse = float(np.sqrt(np.mean((y_true_price - y_pred_price) ** 2)))
    da = float(np.mean(np.sign(y_true_price - today_close) == np.sign(y_pred_price - today_close)) * 100)
    return rmse, da


# ---- walk forward with an expanding window ----
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
rows = []
for k, (tr, te) in enumerate(tscv.split(X), start=1):
    # scale on this fold's training data only
    scaler = StandardScaler().fit(X.iloc[tr])
    Xtr = scaler.transform(X.iloc[tr])
    Xte = scaler.transform(X.iloc[te])

    # tune Lasso honestly inside the training portion only
    lasso = GridSearchCV(Lasso(max_iter=5000), {"alpha": [0.0005, 0.001, 0.01, 0.1]},
                         cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error").fit(Xtr, y_return.iloc[tr])

    # predict the return, then convert back to a price for fair scoring
    pred_ret   = lasso.predict(Xte)
    pred_price = close_t[te] * (1.0 + pred_ret)

    lasso_rmse, lasso_dir = score(price5[te], pred_price, close_t[te])
    naive_rmse, _         = score(price5[te], close_t[te], close_t[te])   # naive = no change

    rows.append({
        "fold": k,
        "test_start": df["Date"].iloc[te[0]].date(),
        "test_end":   df["Date"].iloc[te[-1]].date(),
        "n_test": len(te),
        "Lasso_RMSE": round(lasso_rmse, 2),
        "Naive_RMSE": round(naive_rmse, 2),
        "Lasso_beats_naive": lasso_rmse < naive_rmse,
        "Lasso_Dir%": round(lasso_dir, 1),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))

# ---- summary ----
wins = int(res["Lasso_beats_naive"].sum())
dir_above_50 = int((res["Lasso_Dir%"] > 50).sum())
print("\nSUMMARY across", N_FOLDS, "walk-forward folds:")
print("  folds where Lasso beat naive on RMSE :", wins, "of", N_FOLDS)
print("  folds where direction was above 50%  :", dir_above_50, "of", N_FOLDS)
print("  average directional accuracy         :", round(res["Lasso_Dir%"].mean(), 1), "%")
print("  best / worst fold direction          :", res["Lasso_Dir%"].max(), "% /", res["Lasso_Dir%"].min(), "%")

res.to_csv(os.path.join(OUT_DIR, "walk_forward.csv"), index=False)

# ---- chart: direction per fold vs the 50% coin-flip line ----
plt.figure(figsize=(9, 4))
plt.bar(res["fold"].astype(str), res["Lasso_Dir%"], color="steelblue")
plt.axhline(50, color="red", linestyle="--", label="coin flip (50%)")
plt.ylabel("Directional accuracy %")
plt.xlabel("walk-forward fold (earliest to latest)")
plt.title("Option B (Lasso): directional accuracy per walk-forward fold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "walk_forward.png"))
plt.close()
print("\nsaved walk_forward.csv and walk_forward.png to outputs/")
