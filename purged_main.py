"""
DIG #4: the fully leak-free headline table (purged 80/20, Option B)
=================================================================
The original 80/20 comparison had a tiny boundary leak: the last few training
rows' 5-day labels reached into the test period. Here we PURGE those rows
(train on [0 : split-H]) so the table is completely leak-free, and we add the
honest 'edge over always-up' column.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
try:
    from xgboost import XGBRegressor; HAVE_XGB = True
except Exception: HAVE_XGB = False

H, CORR_DROP, TRAIN_FRAC = 5, 0.95, 0.80

df = pd.read_csv("data/AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
if "Date_col" in df.columns: df = df.drop(columns=["Date_col"])
df["ret"] = df["Close(t)"].shift(-H) / df["Close(t)"] - 1
df["price5"] = df["Close(t)"].shift(-H)
df = df.dropna(subset=["ret"]).reset_index(drop=True)
X = df.drop(columns=["Date", "Close_forcast", "ret", "price5"]).select_dtypes("number")
c = X.corr().abs(); u = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
X = X.drop(columns=[k for k in u.columns if (u[k] > CORR_DROP).any()])
y = df["ret"]; close = df["Close(t)"].values; price5 = df["price5"].values

split = int(len(df) * TRAIN_FRAC)
tr_end = split - H                                     # PURGE the boundary
sc = StandardScaler().fit(X.iloc[:tr_end])
Xtr_s, Xte_s = sc.transform(X.iloc[:tr_end]), sc.transform(X.iloc[split:])
Xtr_r, Xte_r = X.iloc[:tr_end], X.iloc[split:]
ytr = y.iloc[:tr_end]
today = close[split:]; truth = price5[split:]
always_up = (df["ret"].iloc[split:].values > 0).mean() * 100

def score(name, pred_ret):
    pred_price = today * (1 + np.asarray(pred_ret))
    rmse = np.sqrt(mean_squared_error(truth, pred_price))
    dirp = np.mean(np.sign(truth - today) == np.sign(pred_price - today)) * 100
    return {"model": name, "RMSE": round(rmse,2), "Dir%": round(dirp,1),
            "edge_vs_alwaysup": round(dirp - always_up, 1)}

rows = [{"model": "Naive (no change)", "RMSE": round(np.sqrt(mean_squared_error(truth, today)),2),
         "Dir%": 0.0, "edge_vs_alwaysup": None}]
lr = LinearRegression().fit(Xtr_s, ytr); rows.append(score("LinearRegression", lr.predict(Xte_s)))
rows.append(score("Ridge", Ridge(alpha=1.0).fit(Xtr_s, ytr).predict(Xte_s)))
rows.append(score("Lasso", Lasso(alpha=0.001, max_iter=5000).fit(Xtr_s, ytr).predict(Xte_s)))
rows.append(score("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000).fit(Xtr_s, ytr).predict(Xte_s)))
rows.append(score("RandomForest", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(Xtr_r, ytr).predict(Xte_r)))
rows.append(score("AdaBoost", AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42).fit(Xtr_r, ytr).predict(Xte_r)))
if HAVE_XGB:
    rows.append(score("XGBoost", XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42).fit(Xtr_r, ytr).predict(Xte_r)))

print("=" * 66)
print("DIG #4: leak-free headline table (purged 80/20, Option B)")
print("=" * 66)
print(pd.DataFrame(rows).sort_values("RMSE").to_string(index=False))
print(f"\nalways-up base rate on this test set: {always_up:.1f}%")
print("edge_vs_alwaysup near or below 0 = no real directional skill (leak-free).")
