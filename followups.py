"""
Follow-up experiments for Project 1
===================================
Part A: does the train/test split ratio change the result? (70/75/80/90)
Part B: how do we detect overfitting? Compare TRAIN error vs TEST error per model.

All in Option B (predict the 5-day return, score on the price scale), same as the
main pipeline, so the numbers are comparable.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

HORIZON, CORR_DROP = 5, 0.95

# ---- data + return target + features (same recipe) ----
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
y = df["ret"]; close = df["Close(t)"].values; price5 = df["price5"].values

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def dir_acc(true_p, pred_p, today):
    return float(np.mean(np.sign(true_p - today) == np.sign(pred_p - today)) * 100)

# ==========================================================================
# PART A: split-ratio sweep (Option B, Lasso)
# ==========================================================================
print("=" * 74)
print("PART A: does the split ratio matter? (Lasso, Option B)")
print("=" * 74)
rowsA = []
for frac in [0.70, 0.75, 0.80, 0.90]:
    sp = int(len(df) * frac)
    sc = StandardScaler().fit(X.iloc[:sp])
    Xtr, Xte = sc.transform(X.iloc[:sp]), sc.transform(X.iloc[sp:])
    m = GridSearchCV(Lasso(max_iter=5000), {"alpha": [0.0005, 0.001, 0.01, 0.1]},
                     cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error").fit(Xtr, y.iloc[:sp])
    # convert predicted returns to prices, score on price scale
    pred_te = close[sp:] * (1 + m.predict(Xte))
    naive_te = close[sp:]                                    # naive = no change
    rowsA.append({
        "train%": int(frac*100), "n_test": len(df) - sp,
        "Lasso_test_RMSE": round(rmse(price5[sp:], pred_te), 2),
        "naive_test_RMSE": round(rmse(price5[sp:], naive_te), 2),
        "beats_naive": rmse(price5[sp:], pred_te) < rmse(price5[sp:], naive_te),
        "test_Dir%": round(dir_acc(price5[sp:], pred_te, close[sp:]), 1),
    })
print(pd.DataFrame(rowsA).to_string(index=False))

# ==========================================================================
# PART B: overfitting check -- TRAIN error vs TEST error per model
# ==========================================================================
print("\n" + "=" * 74)
print("PART B: overfitting check (80/20). Big TRAIN-vs-TEST gap = overfitting.")
print("=" * 74)
sp = int(len(df) * 0.80)
sc = StandardScaler().fit(X.iloc[:sp])
Xtr_s, Xte_s = sc.transform(X.iloc[:sp]), sc.transform(X.iloc[sp:])
Xtr_r, Xte_r = X.iloc[:sp], X.iloc[sp:]
ytr = y.iloc[:sp]

def score_model(name, fit_pred_tr, fit_pred_te):
    ptr = close[:sp] * (1 + fit_pred_tr)                    # train predictions -> price
    pte = close[sp:] * (1 + fit_pred_te)                    # test predictions -> price
    tr_rmse = rmse(price5[:sp], ptr)
    te_rmse = rmse(price5[sp:], pte)
    return {"model": name, "TRAIN_RMSE": round(tr_rmse, 2), "TEST_RMSE": round(te_rmse, 2),
            "gap_x": round(te_rmse / tr_rmse, 1)}

rowsB = []
lr = LinearRegression().fit(Xtr_s, ytr)
rowsB.append(score_model("LinearRegression", lr.predict(Xtr_s), lr.predict(Xte_s)))
la = Lasso(alpha=0.001, max_iter=5000).fit(Xtr_s, ytr)
rowsB.append(score_model("Lasso", la.predict(Xtr_s), la.predict(Xte_s)))
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(Xtr_r, ytr)
rowsB.append(score_model("RandomForest", rf.predict(Xtr_r), rf.predict(Xte_r)))
if HAVE_XGB:
    xg = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=42).fit(Xtr_r, ytr)
    rowsB.append(score_model("XGBoost", xg.predict(Xtr_r), xg.predict(Xte_r)))
print(pd.DataFrame(rowsB).to_string(index=False))
print("\nRule of thumb: TRAIN_RMSE much smaller than TEST_RMSE (big gap_x) means the")
print("model memorised the training data. gap_x near 1.0 means it generalises honestly.")
