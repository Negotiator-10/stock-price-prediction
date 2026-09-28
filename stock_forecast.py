"""
Stock Price Prediction (AAPL) - complete reference script (heavily commented)
=============================================================================
Project 1 of the IIQF options. This is the FULL, runnable version of the plan
in Stock_Price_Prediction_Plan.md, with every gap filled in AND a plain-English
comment on almost every line explaining the logic and what it does.

Where the same idea repeats (for example plt.close() after each chart), it is
explained the first time and then left alone.

Run it end to end:
    py -3.11 stock_forecast.py

The golden rules this script never breaks:
  - Split by TIME, never shuffle (you only know the past when predicting future).
  - Fit the scaler on TRAIN only (using test data to scale = leakage = fake scores).
  - Drop Close_forcast from inputs (it is the answer; feeding it in is cheating).
  - A model that cannot beat the Naive baseline has no edge. Say so plainly.
"""

# --- imports: pull in the tools we need before we can use them ---------------
import os                       # talk to the file system (make folders, join paths)
import numpy as np              # fast math on arrays; the "np." number toolbox
import pandas as pd             # tables (DataFrames); how we hold the CSV in memory
import matplotlib               # the plotting library (base package)
matplotlib.use("Agg")           # draw charts to files instead of a popup window (headless, safe on any machine)
import matplotlib.pyplot as plt # the drawing commands live under "plt."

# seaborn makes a prettier heatmap, but it is optional. Try to import it;
# if it is not installed, set a flag so we can fall back later instead of crashing.
try:
    import seaborn as sns       # attempt the import
    HAVE_SEABORN = True         # success: remember we have it
except Exception:
    HAVE_SEABORN = False        # failure: remember we do NOT have it

# scikit-learn pieces: preprocessing, the linear models, tuning tools, tree models, metrics.
from sklearn.preprocessing import StandardScaler                       # rescales features to mean 0, std 1
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # the four linear models
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV      # time-aware cross-validation + hyperparameter search
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor  # two tree ensemble models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # error scores

# XGBoost is a separate install. Same optional-import pattern as seaborn.
try:
    from xgboost import XGBRegressor   # the gradient-boosted trees model
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# TensorFlow / Keras is the heaviest dependency (the neural net). Also optional.
try:
    from tensorflow.keras.models import Sequential            # a straight stack of layers
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout  # the layer types we use
    HAVE_TF = True
except Exception:
    HAVE_TF = False


# ---------------------------------------------------------------------------
# Config: every setting you might change lives here in ONE place.
# ---------------------------------------------------------------------------
DATA_PATH = r"data/AAPL.csv"  # where the CSV is (r"" = raw string so backslashes are literal)
OUT_DIR   = r"outputs"        # where charts and the results table get saved
HORIZON   = 5      # how many trading days ahead we predict (5 = about one week)
LOOKBACK  = 10     # how many past days the LSTM/GRU reads as one window
CORR_DROP = 0.95   # if two features correlate above this, drop one as redundant
TRAIN_FRAC = 0.80  # fraction of the timeline used for training (oldest 80 percent)

# OPTION A vs OPTION B: what the models are asked to predict.
#   RETURN_MODE = False -> Option A: predict the raw price HORIZON days ahead.
#   RETURN_MODE = True  -> Option B: predict the HORIZON-day RETURN, then convert it back to a price.
# Either way we always SCORE on the price scale, so the two runs are directly comparable and both
# face the same naive baseline. Flip this one flag to switch experiments.
RETURN_MODE = True
SUFFIX = "_optionB" if RETURN_MODE else ""   # keeps Option A and Option B output files separate

os.makedirs(OUT_DIR, exist_ok=True)   # create the outputs folder now; exist_ok=True means "fine if it already exists"


def banner(text):
    """Print a clear step header. Keeps ASCII only, so the Windows console never chokes on odd characters."""
    line = "=" * 70                    # a row of 70 equals signs
    print("\n" + line)                 # blank line then the top rule
    print(text)                        # the step title
    print(line)                        # the bottom rule


# ===========================================================================
# STEP 1. Import and order the data
# ===========================================================================
banner("STEP 1  Load and sort by date")

df = pd.read_csv(DATA_PATH)                                   # read the CSV file into a table called df
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")   # turn the Date text column into real dates (day-month-year layout)
df = df.sort_values("Date").reset_index(drop=True)           # sort oldest to newest, then renumber rows 0,1,2,... cleanly
print("shape:", df.shape)                                    # (rows, columns); expect (3732, 64)
print("range:", df["Date"].min().date(), "to", df["Date"].max().date())  # earliest and latest dates, as a sanity check


# ===========================================================================
# STEP 2. Quick clean
# ===========================================================================
banner("STEP 2  Clean check and drop duplicate date column")

print("total missing cells:", int(df.isna().sum().sum()))    # count all empty cells across the whole table; expect 0
if "Date_col" in df.columns:                                 # only act if that duplicate column is actually present
    df = df.drop(columns=["Date_col"])                       # remove Date_col (it just repeats Date)
    print("dropped Date_col")


# ===========================================================================
# STEP 3. Explore (EDA)  - draw the price path and the correlation heatmap
# ===========================================================================
banner("STEP 3  EDA charts")

df.set_index("Date")["Close(t)"].plot(figsize=(11, 4),       # plot closing price with Date on the x-axis; figsize is width,height in inches
                                       title="AAPL close 2005 to 2020")
plt.tight_layout()                                           # tidy the spacing so labels are not cut off
plt.savefig(os.path.join(OUT_DIR, "price.png"))              # write the chart to outputs/price.png
plt.close()                                                  # free the figure from memory (do this after every chart)

corr = df.corr(numeric_only=True)                            # correlation of every numeric column with every other (-1 to 1)
plt.figure(figsize=(14, 12))                                 # start a fresh, large square canvas for the heatmap
if HAVE_SEABORN:                                             # if seaborn is available, use its nicer heatmap
    sns.heatmap(corr, cmap="coolwarm", center=0)             # blue-white-red colours, centered on 0
else:                                                        # otherwise fall back to plain matplotlib
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)       # draw the matrix as an image, colour range fixed to -1..1
    plt.colorbar()                                           # add the colour scale legend
plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "corr_heatmap.png"))
plt.close()
print("saved price.png and corr_heatmap.png to outputs/")


# ===========================================================================
# STEP 4. Set the target (five trading days ahead = about one week)
# ===========================================================================
banner("STEP 4  Build the 5-day-ahead target")

# Build BOTH targets, so we can LEARN from whichever the mode picks but always SCORE on price.
#   target_price  = the actual close HORIZON days ahead (Option A). shift(-HORIZON) pulls a future close back to today's row.
#   target_return = the percentage change over the next HORIZON days (Option B). Does not trend, so it is a fairer target.
df["target_price"]  = df["Close(t)"].shift(-HORIZON)
df["target_return"] = df["Close(t)"].shift(-HORIZON) / df["Close(t)"] - 1

# the learning target the models actually see: return in Option B, raw price in Option A
df["target"] = df["target_return"] if RETURN_MODE else df["target_price"]

# The last HORIZON rows look into a future that does not exist yet, so their targets are blank. Drop them.
df = df.dropna(subset=["target_price"]).reset_index(drop=True)  # drop rows with no known future, then renumber
print("mode:", "Option B (predict 5-day RETURN)" if RETURN_MODE else "Option A (predict raw price)")
print("rows after building target:", len(df))


# ===========================================================================
# STEP 5. Choose features and cut redundancy
# ===========================================================================
banner("STEP 5  Feature matrix X, target y, drop near-duplicate features")

# These columns must NOT be inputs: Date (not a number), Close_forcast (the leaky provided answer),
# and the three target columns we just built. Everything else is fair game, including Close(t) which is known today.
leak_or_meta = ["Date", "Close_forcast", "target", "target_price", "target_return"]
X = df.drop(columns=[c for c in leak_or_meta if c in df.columns])  # X = all columns except those three (guarded in case one is absent)
y = df["target"]                                                   # y = the answer column on its own

X = X.select_dtypes(include=[np.number])   # keep only numeric columns, in case any text column slipped through

# Multicollinearity cleanup: many indicators say the same thing (MA5, MA10, MA20 all track recent price).
cmat = X.corr().abs()                                              # correlation between every feature pair, absolute value (strength, ignore sign)
upper = cmat.where(np.triu(np.ones(cmat.shape), k=1).astype(bool)) # keep only the upper triangle so each pair is checked once, not twice
to_drop = [c for c in upper.columns if (upper[c] > CORR_DROP).any()]  # list any column that is >0.95 correlated with an earlier one
X = X.drop(columns=to_drop)                                        # remove those redundant columns
print("dropped", len(to_drop), "redundant features; kept", X.shape[1])
print("kept columns:", list(X.columns))


# ===========================================================================
# STEP 6. Split by time, never randomly
# ===========================================================================
banner("STEP 6  Time-ordered train / test split")

split = int(len(df) * TRAIN_FRAC)          # the row index where train ends and test begins (80 percent point)
X_train, X_test = X.iloc[:split], X.iloc[split:]   # features: first 80 percent train, last 20 percent test
y_train, y_test = y.iloc[:split], y.iloc[split:]   # answers: same split, kept in lockstep with the features

# We also grab today's close on the TEST rows. Needed twice later:
#   1) the Naive baseline (predict "no change from today"),
#   2) directional accuracy (did the price go up or down vs today?).
today_close_test = df["Close(t)"].iloc[split:].values   # .values gives a plain numpy array

# the actual future PRICE on the test rows. We always score against this, whatever the mode,
# so Option A and Option B land on one common dollar scale and share the same naive baseline.
y_test_price = df["target_price"].iloc[split:].values
print("train:", X_train.shape, " test:", X_test.shape)


# ===========================================================================
# STEP 7. Scale the features (fit on TRAIN only)
# ===========================================================================
banner("STEP 7  Scale (fit on train only)")

scaler = StandardScaler().fit(X_train)     # LEARN each feature's mean and spread FROM TRAINING DATA ONLY
X_train_s = scaler.transform(X_train)      # apply that scaling to train
X_test_s  = scaler.transform(X_test)       # apply the SAME train-based scaling to test (no peeking at test stats)
print("scaled. train mean ~0:", round(float(X_train_s.mean()), 6))  # sanity check: scaled train data centers near 0


# ---------------------------------------------------------------------------
# One shared bucket for every model's test-set predictions, so Step 12 can
# score them all in a single loop. Rule of thumb:
#   linear models + neural net -> use the SCALED X
#   tree models                -> use the raw UNSCALED X (splits do not care about scale)
# ---------------------------------------------------------------------------
predictions = {}   # empty dict; we will fill it with name -> prediction array


# ===========================================================================
# STEP 8. Baseline model: Simple Linear Regression
# ===========================================================================
banner("STEP 8  LinearRegression")

lr = LinearRegression().fit(X_train_s, y_train)   # fit a straight-line model on scaled train features
predictions["LinearRegression"] = lr.predict(X_test_s)  # predict the test set and stash it under its name
print("done")


# ===========================================================================
# STEP 9. Regularised linear models: Ridge, Lasso, ElasticNet
# ===========================================================================
banner("STEP 9  Ridge, Lasso, ElasticNet (time-aware tuning)")

tscv = TimeSeriesSplit(n_splits=5)   # cross-validation that always trains on the past and validates on the future (5 folds)

# GridSearchCV tries every listed alpha, scores each with time-aware CV, and keeps the best.
# alpha = regularisation strength: higher = simpler model, less overfitting.
ridge = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]},          # Ridge shrinks all weights (tames correlated features)
                     cv=tscv, scoring="neg_root_mean_squared_error"        # score = -RMSE (higher is better, so less error wins)
                     ).fit(X_train_s, y_train)

lasso = GridSearchCV(Lasso(max_iter=5000), {"alpha": [0.001, 0.01, 0.1, 1]},  # Lasso can zero out weak features entirely (selection)
                     cv=tscv, scoring="neg_root_mean_squared_error"           # max_iter=5000 gives it enough steps to converge
                     ).fit(X_train_s, y_train)

enet = GridSearchCV(ElasticNet(max_iter=5000),                                # ElasticNet blends Ridge and Lasso behaviour
                    {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]},  # l1_ratio tunes the blend (0=Ridge-like, 1=Lasso-like)
                    cv=tscv, scoring="neg_root_mean_squared_error"
                    ).fit(X_train_s, y_train)

predictions["Ridge"]      = ridge.predict(X_test_s)   # best Ridge predicts test
predictions["Lasso"]      = lasso.predict(X_test_s)   # best Lasso predicts test
predictions["ElasticNet"] = enet.predict(X_test_s)    # best ElasticNet predicts test
print("best alphas:", ridge.best_params_, lasso.best_params_, enet.best_params_)  # show which settings won


# ===========================================================================
# STEP 10. Tree ensembles: Random Forest, XGBoost, AdaBoost (UNSCALED)
# ===========================================================================
banner("STEP 10  RandomForest, XGBoost, AdaBoost")

rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1   # 400 trees; random_state fixes randomness for repeatability; n_jobs=-1 uses all CPU cores
                           ).fit(X_train, y_train)                        # note: RAW X_train, not scaled
predictions["RandomForest"] = rf.predict(X_test)

ada = AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42   # 300 weak learners; small learning_rate = slower, steadier boosting
                        ).fit(X_train, y_train)
predictions["AdaBoost"] = ada.predict(X_test)

if HAVE_XGB:                                                              # only if xgboost imported successfully
    xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, # 500 trees, each shallow (depth 4) to avoid overfitting
                       subsample=0.8, colsample_bytree=0.8, random_state=42  # each tree sees 80% of rows and 80% of features (adds robustness)
                       ).fit(X_train, y_train)
    predictions["XGBoost"] = xgb.predict(X_test)

    imp = pd.Series(xgb.feature_importances_, index=X_train.columns       # pair each feature with how much XGBoost relied on it
                    ).sort_values(ascending=False)                        # sort most important first
    print("top 15 features by XGBoost importance:")
    print(imp.head(15).to_string())                                       # show the top 15
else:
    print("xgboost not installed, skipping. (pip install xgboost)")


# ===========================================================================
# STEP 11. Recurrent net: LSTM or GRU (SCALED, 3D windows)
# ===========================================================================
banner("STEP 11  LSTM / GRU")

def make_sequences(Xs, y_series, lookback):
    """Turn a flat 2D array of rows into 3D sliding windows for a recurrent net.

    A recurrent net reads a WINDOW of the last `lookback` days at once, not a single row.
    The first `lookback` rows cannot form a full window, so the output is SHORTER than
    the input by exactly `lookback` rows. We must remember that when we score it later,
    or we will line up the wrong answers with the wrong predictions.
    """
    Xseq, yseq = [], []                          # empty lists to collect windows and their answers
    for i in range(lookback, len(Xs)):           # start at row `lookback` (first row with enough history behind it)
        Xseq.append(Xs[i - lookback:i])          # the window: rows [i-lookback .. i-1], shape (lookback, n_features)
        yseq.append(y_series.iloc[i])            # the answer that goes with that window: target at row i
    return np.array(Xseq), np.array(yseq)        # stack the lists into arrays and hand them back

USE_GRU = False   # False = build an LSTM; flip to True to build a GRU instead (a lighter cousin)

if HAVE_TF:                                                      # only run this whole block if TensorFlow is installed
    Xtr_seq, ytr_seq = make_sequences(X_train_s, y_train, LOOKBACK)  # windows for training (from scaled train)
    Xte_seq, yte_seq = make_sequences(X_test_s,  y_test,  LOOKBACK)  # windows for testing (from scaled test)

    today_close_seq = today_close_test[LOOKBACK:]   # trim today's-close the same way (first LOOKBACK rows are gone), so it stays aligned
    y_test_price_seq = y_test_price[LOOKBACK:]      # actual future PRICE aligned to the sequence test set, for scoring

    # pick the recurrent layer type based on the toggle above. 32 = number of memory units.
    recurrent_layer = (GRU(32, input_shape=(LOOKBACK, X_train_s.shape[1]))   # input_shape = (timesteps per window, features per day)
                       if USE_GRU else
                       LSTM(32, input_shape=(LOOKBACK, X_train_s.shape[1])))

    net = Sequential([recurrent_layer,   # layer 1: the LSTM/GRU that reads the sequence
                      Dropout(0.2),      # layer 2: randomly zero 20% of connections while training, to fight overfitting
                      Dense(1)])         # layer 3: one output neuron = the single predicted price
    net.compile(optimizer="adam", loss="mse")   # adam = the training algorithm; mse = the error it minimises
    net.fit(Xtr_seq, ytr_seq, epochs=30, batch_size=32, verbose=0)  # train 30 passes over the data, 32 windows at a time, silently

    pred_net = net.predict(Xte_seq, verbose=0).ravel()  # predict test windows; .ravel() flattens the (n,1) output to a flat list
    net_name = "GRU" if USE_GRU else "LSTM"             # a label for the results table
    print(net_name, "trained. sequence test length:", len(pred_net))
else:
    # If TF is missing, set these to None so Step 12 knows to skip the net cleanly.
    pred_net = None
    yte_seq = None
    today_close_seq = None
    y_test_price_seq = None
    net_name = None
    print("tensorflow not installed, skipping the recurrent net. (pip install tensorflow)")


# ===========================================================================
# STEP 12. Evaluate every model the same way
# ===========================================================================
banner("STEP 12  Evaluate against the naive baseline")

def evaluate(name, y_true, y_pred, today_close):
    """Score one model. Returns a dict of metrics so we can build a table of them."""
    y_true = np.asarray(y_true, dtype=float)     # force the true answers into a numeric array
    y_pred = np.asarray(y_pred, dtype=float)     # force the predictions into a numeric array
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # typical error in dollars; squares punish big misses harder
    mae  = mean_absolute_error(y_true, y_pred)           # average error in dollars, treating all misses equally
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # average error as a percentage of the true price
    r2   = r2_score(y_true, y_pred)                      # fraction of price movement explained (1 is perfect, 0 is useless)
    # directional accuracy: of all days, how often did we call up-vs-down correctly relative to today's price?
    da = np.mean(np.sign(y_true - today_close) == np.sign(y_pred - today_close)) * 100  # sign(+/-) match rate, as a percent
    return {"model": name, "RMSE": rmse, "MAE": mae, "MAPE%": mape, "R2": r2, "Dir%": da}

def to_price(pred, today_close):
    """Turn a model's raw output into a PRICE so every model is scored on one scale.
    In Option B the output is a return, so price = today * (1 + return).
    In Option A the output already IS a price, so it passes straight through."""
    pred = np.asarray(pred, dtype=float)
    return today_close * (1.0 + pred) if RETURN_MODE else pred

rows = []   # collect one metrics dict per model here

# The baseline first: "the price 5 days from now equals today's price." Every model must beat this.
# Naive is a no-change price call in both modes, so it is unchanged.
rows.append(evaluate("Naive (no change)", y_test_price, today_close_test, today_close_test))

# Score every non-sequence model on the identical full test set. Convert to price first (no-op in Option A).
for name, pred in predictions.items():
    rows.append(evaluate(name, y_test_price, to_price(pred, today_close_test), today_close_test))

# Score the recurrent net separately, because its test set is shorter (windows) and specially aligned.
if pred_net is not None:
    rows.append(evaluate(net_name, y_test_price_seq, to_price(pred_net, today_close_seq), today_close_seq))

results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)  # turn the list of dicts into a table, best RMSE on top
print(results.round(3).to_string(index=False))                          # print it tidily, 3 decimals, no row numbers
print("\nReminder: beating Naive on RMSE is the bar. A good RMSE with Dir% "
      "near 50 is useless for trading.")


# ===========================================================================
# STEP 13. Compare and pick the winner
# ===========================================================================
banner("STEP 13  Save comparison and pick the winner")

results.to_csv(os.path.join(OUT_DIR, "model_comparison%s.csv" % SUFFIX), index=False)  # save the table (separate file per mode)

naive_rmse = float(results.loc[results["model"] == "Naive (no change)", "RMSE"].iloc[0])  # pull out the Naive RMSE number
ranked = results[results["model"] != "Naive (no change)"]  # everything except Naive (Naive is the bar, not a candidate)
best_row = ranked.iloc[0]                                   # top row = smallest RMSE (table is already sorted)
best_name = best_row["model"]                              # the winning model's name
beats_naive = best_row["RMSE"] < naive_rmse                # True only if the winner actually has less error than Naive

print("best model:", best_name, " RMSE:", round(float(best_row["RMSE"]), 3))
print("beats naive baseline:", bool(beats_naive))
if not beats_naive:                                        # be honest if nothing worked
    print("HONEST NOTE: no model beat the naive baseline. Report that as the result.")


# ===========================================================================
# STEP 14. Forecast the next week with the chosen model
# ===========================================================================
banner("STEP 14  Forecast " + str(HORIZON) + " trading days ahead")

# Look up the winning name to its fitted object. Two groups because they take different input:
scaled_models = {                                          # these need SCALED input
    "LinearRegression": lr, "Ridge": ridge, "Lasso": lasso, "ElasticNet": enet,
}
unscaled_models = {                                        # these take RAW input
    "RandomForest": rf, "AdaBoost": ada,
}
if HAVE_XGB:
    unscaled_models["XGBoost"] = xgb                       # add XGBoost only if it exists

latest = X.iloc[[-1]]   # the single newest row of features (double brackets keep it a 1-row DataFrame, not a Series)

if best_name in scaled_models:                             # winner is a linear model
    model = scaled_models[best_name]
    forecast = float(model.predict(scaler.transform(latest))[0])   # scale the row, predict, take the one number out
elif best_name in unscaled_models:                        # winner is a tree model
    model = unscaled_models[best_name]
    forecast = float(model.predict(latest)[0])                     # predict directly on the raw row
elif best_name in ("LSTM", "GRU") and HAVE_TF:            # winner is the neural net
    last_window = scaler.transform(X.iloc[-LOOKBACK:]).reshape(1, LOOKBACK, X.shape[1])  # last LOOKBACK rows, scaled, shaped as one window
    forecast = float(net.predict(last_window, verbose=0).ravel()[0])
else:                                                      # safety net: if something odd happened, fall back to plain linear
    model = lr
    forecast = float(lr.predict(scaler.transform(latest))[0])
    best_name = "LinearRegression (fallback)"

# Honest uncertainty: quote the winner's own test RMSE as a plus/minus band, never a bare single number.
band = float(best_row["RMSE"])                            # typical error in dollars (we scored on the price scale)
last_close = float(df["Close(t)"].iloc[-1])              # today's actual close, for context

# In Option B the raw output is a return, so convert it to a price. In Option A it already is a price.
forecast_price = float(last_close * (1.0 + forecast)) if RETURN_MODE else float(forecast)

print("model used     :", best_name)
if RETURN_MODE:
    print("predicted return:", round(forecast * 100, 2), "% over", HORIZON, "days")
print("today's close  :", round(last_close, 2))
print("forecast (t+%d):" % HORIZON, round(forecast_price, 2))
print("uncertainty    : about +/-", round(band, 2), "dollars (one test RMSE)")

# Draw a small chart: the last 120 days of price, plus the forecast point with its error bar.
recent = df.set_index("Date")["Close(t)"].iloc[-120:]    # last 120 closes, indexed by date
plt.figure(figsize=(11, 4))                              # fresh canvas
plt.plot(recent.index, recent.values, label="Close(t)") # the recent price line
future_x = recent.index[-1] + pd.tseries.offsets.BusinessDay(HORIZON)  # a date HORIZON business days after the last one, for the forecast dot
plt.errorbar(future_x, forecast_price, yerr=band, fmt="o", color="red",   # the forecast as a red dot with a vertical +/- band
             capsize=5, label="forecast t+%d" % HORIZON)
mode_label = "Option B, return" if RETURN_MODE else "Option A, price"
plt.title("AAPL: recent close and %d-day forecast (%s, %s)" % (HORIZON, best_name, mode_label))
plt.legend()                                             # show the labels box
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "forecast%s.png" % SUFFIX))
plt.close()
print("saved forecast%s.png to outputs/" % SUFFIX)


# ===========================================================================
# STEP 15. Write up (done by hand in the report / docx)
# ===========================================================================
banner("DONE")
print("Deliverables written to:", OUT_DIR)
print(" - price.png, corr_heatmap.png, model_comparison%s.csv, forecast%s.png" % (SUFFIX, SUFFIX))
print("Mode was:", "Option B (5-day return)" if RETURN_MODE else "Option A (raw price)")
