"""
DIG #1: can we predict VOLATILITY (how big the move is), not direction?
======================================================================
Direction was unpredictable. Volatility should be different: calm clusters with
calm, storms with storms (volatility clustering, the basis of GARCH). We predict
next-5-day realised volatility and test HONESTLY (purged, vs proper baselines).

Target: realised vol over the next 5 days = std of the next 5 daily returns.
Baselines:
  - constant  : always predict the average vol (the no-skill floor).
  - persistence: predict future vol = recent trailing vol (the honest bar,
                 this is what volatility clustering alone gives you).
The real question: is vol predictable at all (R2 >> 0), and can the model add
anything beyond simple persistence?
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
try:
    from xgboost import XGBRegressor; HAVE_XGB = True
except Exception: HAVE_XGB = False

H, CORR_DROP, TRAIN_FRAC = 5, 0.95, 0.80

df = pd.read_csv("data/AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns: df = df.drop(columns=["Date_col"])
daily = df["Close(t)"].pct_change()

# target: std of the NEXT H daily returns (future realised vol)
fut = pd.concat([daily.shift(-k) for k in range(1, H+1)], axis=1)
df["fut_vol"] = fut.std(axis=1)
# persistence predictor: std of the LAST H daily returns (past only)
past = pd.concat([daily.shift(k) for k in range(0, H)], axis=1)
df["trail_vol5"] = past.std(axis=1)
df["trail_vol20"] = daily.rolling(20).std()

df = df.dropna(subset=["fut_vol", "trail_vol5", "trail_vol20"]).reset_index(drop=True)

# features: existing set (already contains ATR, SD20, STD5 etc.)
X = df.drop(columns=["Date", "Close_forcast", "fut_vol"]).select_dtypes("number")
X = X.drop(columns=[c for c in ["trail_vol5", "trail_vol20"] if c in X.columns])  # keep baselines separate
c = X.corr().abs(); up = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
X = X.drop(columns=[k for k in up.columns if (up[k] > CORR_DROP).any()])
y = df["fut_vol"]

# purged 80/20 split (drop last H training rows so no future leaks in)
split = int(len(df) * TRAIN_FRAC)
tr_end = split - H
sc = StandardScaler().fit(X.iloc[:tr_end])
Xtr, Xte = sc.transform(X.iloc[:tr_end]), sc.transform(X.iloc[split:])
ytr, yte = y.iloc[:tr_end], y.iloc[split:]

print("=" * 70)
print("DIG #1: predicting 5-day realised VOLATILITY (purged 80/20)")
print("=" * 70)

# baselines on the test set
const_pred = np.full(len(yte), ytr.mean())
r2_const = r2_score(yte, const_pred)
r2_persist5  = r2_score(yte, df["trail_vol5"].iloc[split:])
r2_persist20 = r2_score(yte, df["trail_vol20"].iloc[split:])

print(f"\nBaselines (R2 on test):")
print(f"  constant (mean vol)          : {r2_const:6.3f}   <- the no-skill floor")
print(f"  persistence, last 5-day vol  : {r2_persist5:6.3f}")
print(f"  persistence, last 20-day vol : {r2_persist20:6.3f}   <- the honest bar")

# collect every row (name, kind, R2) so the result is saved, not just printed
rows = [
    {"name": "constant (mean vol)",          "kind": "baseline", "R2": r2_const},
    {"name": "persistence, last 5-day vol",  "kind": "baseline", "R2": r2_persist5},
    {"name": "persistence, last 20-day vol", "kind": "baseline", "R2": r2_persist20},
]

print(f"\nModels (R2 on test):")
lr = LinearRegression().fit(Xtr, ytr)
r2_lr = r2_score(yte, lr.predict(Xte))
print(f"  LinearRegression : {r2_lr:6.3f}")
rows.append({"name": "LinearRegression", "kind": "model", "R2": r2_lr})
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(X.iloc[:tr_end], ytr)
r2_rf = r2_score(yte, rf.predict(X.iloc[split:]))
print(f"  RandomForest     : {r2_rf:6.3f}")
rows.append({"name": "RandomForest", "kind": "model", "R2": r2_rf})
if HAVE_XGB:
    xg = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=42).fit(X.iloc[:tr_end], ytr)
    r2_xg = r2_score(yte, xg.predict(X.iloc[split:]))
    print(f"  XGBoost          : {r2_xg:6.3f}")
    rows.append({"name": "XGBoost", "kind": "model", "R2": r2_xg})

corr = np.corrcoef(yte, lr.predict(Xte))[0, 1]
print(f"\nCorrelation(predicted vol, actual vol), LinearRegression: {corr:.3f}")
print("\nContrast: predicting RETURNS gave R2 near 0 or negative (no skill).")
print("If these vol R2 values are strongly positive, volatility IS predictable.")

# save results so this dig has a matching file in outputs, like the other models
out = pd.DataFrame(rows)
out["R2"] = out["R2"].round(3)
out.to_csv("outputs/volatility_comparison.csv", index=False)
print("\nSaved outputs/volatility_comparison.csv")
