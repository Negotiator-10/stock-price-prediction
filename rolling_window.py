"""
Rolling-window experiment (follow-up 2)
=======================================
Question: is it better to train on only recent data (a short window) or on all
history? We walk forward through the SAME test period and, at each step, retrain
the Lasso on different amounts of recent history:

    63 days  (~3 months)
    252 days (~1 year)
    504 days (~2 years)
    expanding (all history up to that point)

Every method predicts the SAME set of days (starting after day 504), so the
comparison is fair. We report RMSE (price scale) vs naive, and directional
accuracy, for each window size.

Fixed Lasso alpha (no grid search) so the many refits run quickly.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

HORIZON, CORR_DROP, STEP, START = 5, 0.95, 5, 504

df = pd.read_csv("data/AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns: df = df.drop(columns=["Date_col"])
df["ret"] = df["Close(t)"].shift(-HORIZON) / df["Close(t)"] - 1
df["price5"] = df["Close(t)"].shift(-HORIZON)
df = df.dropna(subset=["ret"]).reset_index(drop=True)
X = df.drop(columns=["Date", "Close_forcast", "ret", "price5"]).select_dtypes("number")
c = X.corr().abs(); up = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
X = X.drop(columns=[k for k in up.columns if (up[k] > CORR_DROP).any()])
Xv = X.values
y = df["ret"].values; close = df["Close(t)"].values; price5 = df["price5"].values

def walk(window):
    """window = int (rolling) or None (expanding). Returns (rmse, naive_rmse, dir%)."""
    preds, trues, todays = [], [], []
    for t in range(START, len(df) - HORIZON, STEP):
        lo = 0 if window is None else max(0, t - window)
        Xtr, ytr = Xv[lo:t], y[lo:t]
        sc = StandardScaler().fit(Xtr)
        m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(Xtr), ytr)
        pr = m.predict(sc.transform(Xv[t:t+1]))[0]
        preds.append(close[t] * (1 + pr)); trues.append(price5[t]); todays.append(close[t])
    preds, trues, todays = map(np.array, (preds, trues, todays))
    rmse = np.sqrt(np.mean((trues - preds) ** 2))
    naive = np.sqrt(np.mean((trues - todays) ** 2))
    da = np.mean(np.sign(trues - todays) == np.sign(preds - todays)) * 100
    return rmse, naive, da, len(preds)

print("=" * 70)
print("ROLLING-WINDOW EXPERIMENT (same test days for every window)")
print("=" * 70)
rows = []
for name, w in [("63d (~3mo)", 63), ("252d (~1yr)", 252), ("504d (~2yr)", 504), ("expanding (all)", None)]:
    rmse, naive, da, n = walk(w)
    rows.append({"train_window": name, "n_preds": n, "RMSE": round(rmse, 2),
                 "naive_RMSE": round(naive, 2), "beats_naive": rmse < naive,
                 "Dir%": round(da, 1)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nReading it: a very short window (63d) has too little data for 24 features,")
print("so expect noisier, weaker results. Longer/expanding windows are steadier.")
