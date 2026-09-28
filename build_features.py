"""
Build my OWN feature sheet for AAPL, from the raw prices only
=============================================================
Goal: take just the RAW data from the provided AAPL.csv (Open, High, Low, Close,
Volume, and the three index closes QQQ / S&P / Dow) and rebuild every indicator
myself with formulas I control, then save it as an Excel file.

Why do this:
  - I understand every single column, because I wrote the formula for it.
  - It uses the exact same dates and prices as the provided sheet (fully offline).
  - It adds both targets ready to model: the 5-day-ahead price (Option A) and the
    5-day forward return (Option B).

Run:
    .venv\\Scripts\\python.exe build_features.py

Output:
    AAPL_my_features.xlsx   (my transparent, self-built version)

Note on the raw prices: these are the SAME split-adjusted prices as the provided
file (close 6.60 in 2005, 460 in 2020). If you instead downloaded AAPL from Yahoo
today, every price would also be adjusted for the Aug 2020 four-for-one split, so
it would be about four times smaller and would not match. We deliberately reuse
their raw prices so our sheet lines up with theirs.
"""

import numpy as np
import pandas as pd

SRC = r"data/AAPL.csv"
OUT = r"AAPL_my_features.xlsx"
HORIZON = 5   # trading days ahead for the target (about one week)


# ---------------------------------------------------------------------------
# 1. Load ONLY the raw columns. Everything else we will compute ourselves.
# ---------------------------------------------------------------------------
raw = pd.read_csv(SRC)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%Y")
raw = raw.sort_values("Date").reset_index(drop=True)

# start a fresh table with just the true source data
df = pd.DataFrame()
df["Date"]   = raw["Date"]
df["Open"]   = raw["Open"]
df["High"]   = raw["High"]
df["Low"]    = raw["Low"]
df["Close"]  = raw["Close(t)"]
df["Volume"] = raw["Volume"]
# the three market indices are also raw source data
df["QQQ_Close"]  = raw["QQQ_Close"]
df["SnP_Close"]  = raw["SnP_Close"]
df["DJIA_Close"] = raw["DJIA_Close"]

C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]


# ---------------------------------------------------------------------------
# Small helper formulas, written once so the code below reads cleanly.
# min_periods=1 means early rows still get a value (using the days available so
# far) instead of a blank, which keeps every date populated like the provided
# sheet. The earliest rows are therefore "settling" and use fewer days than
# their name suggests. That is the honest trade for having no blank cells.
# ---------------------------------------------------------------------------
def sma(series, n):                       # simple moving average
    return series.rolling(n, min_periods=1).mean()

def ema(series, n):                       # exponential moving average (recent days weigh more)
    return series.ewm(span=n, adjust=False).mean()

def rolling_std(series, n):               # rolling standard deviation (a volatility gauge)
    return series.rolling(n, min_periods=1).std()

def wilder(series, n):                    # Wilder smoothing, the classic average used by RSI/ATR/ADX
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


# ---------------------------------------------------------------------------
# 2. Trend: simple and exponential moving averages
# ---------------------------------------------------------------------------
for n in (5, 10, 20, 50, 200):
    df[f"MA{n}"] = sma(C, n)              # average close over the last n days
for n in (10, 20, 50, 100, 200):
    df[f"EMA{n}"] = ema(C, n)            # same idea, but reacts faster to recent price


# ---------------------------------------------------------------------------
# 3. Volatility: rolling std, Bollinger Bands, Average True Range
# ---------------------------------------------------------------------------
df["SD20"] = rolling_std(C, 20)                          # 20-day price standard deviation
df["STD5"] = rolling_std(C, 5)                           # 5-day version
df["Upper_Band"] = df["MA20"] + 2 * df["SD20"]           # Bollinger upper = MA20 plus two std
df["Lower_Band"] = df["MA20"] - 2 * df["SD20"]           # Bollinger lower = MA20 minus two std

# True Range = the biggest of: today's range, or the gap from yesterday's close.
prev_close = C.shift(1)
true_range = pd.concat([
    H - L,                       # today's high minus low
    (H - prev_close).abs(),      # high vs yesterday's close (catches gaps up)
    (L - prev_close).abs(),      # low vs yesterday's close (catches gaps down)
], axis=1).max(axis=1)
df["ATR14"] = wilder(true_range, 14)                     # Average True Range, a pure volatility measure


# ---------------------------------------------------------------------------
# 4. Momentum: MACD, RSI, ROC, Williams %R, Stochastic %K, CCI
# ---------------------------------------------------------------------------
# MACD = fast EMA minus slow EMA; the signal line is a smoothed MACD.
df["MACD"] = ema(C, 12) - ema(C, 26)                     # positive = short-term momentum above long-term
df["MACD_signal"] = ema(df["MACD"], 9)                   # the trigger line

# RSI(14): compares average up-moves to average down-moves, scaled 0 to 100.
delta = C.diff()                                         # day-over-day price change
gain = delta.clip(lower=0)                               # keep only the ups
loss = -delta.clip(upper=0)                              # keep only the downs (as positive numbers)
rs = wilder(gain, 14) / wilder(loss, 14)                # ratio of average gain to average loss
df["RSI14"] = 100 - 100 / (1 + rs)                       # above 70 overbought, below 30 oversold

# ROC: percentage change over the last 12 days, raw momentum.
df["ROC12"] = (C / C.shift(12) - 1) * 100

# Williams %R(14): where today's close sits in the recent high-low range, scaled -100 to 0.
hh14 = H.rolling(14, min_periods=1).max()                # highest high over 14 days
ll14 = L.rolling(14, min_periods=1).min()                # lowest low over 14 days
df["WilliamsR14"] = (hh14 - C) / (hh14 - ll14) * -100

# Stochastic %K(14): the same range idea, scaled 0 to 100.
df["StochK14"] = (C - ll14) / (hh14 - ll14) * 100

# CCI(20): how far price is from its recent average, in units of mean deviation.
tp = (H + L + C) / 3                                      # typical price
sma_tp = sma(tp, 20)
mean_dev = (tp - sma_tp).abs().rolling(20, min_periods=1).mean()
df["CCI20"] = (tp - sma_tp) / (0.015 * mean_dev)


# ---------------------------------------------------------------------------
# 5. Trend strength: ADX(14)
# ---------------------------------------------------------------------------
# ADX measures how STRONG a trend is, ignoring its direction.
up_move = H.diff()                                        # today's high minus yesterday's high
down_move = -L.diff()                                     # yesterday's low minus today's low
plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)    # strong up pressure
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)  # strong down pressure
atr_for_adx = wilder(true_range, 14)
plus_di  = 100 * wilder(pd.Series(plus_dm,  index=df.index), 14) / atr_for_adx
minus_di = 100 * wilder(pd.Series(minus_dm, index=df.index), 14) / atr_for_adx
dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100   # directional index
df["ADX14"] = wilder(dx, 14)                             # smoothed = ADX; high means a strong trend


# ---------------------------------------------------------------------------
# 6. Volume based: Force Index
# ---------------------------------------------------------------------------
df["ForceIndex1"]  = (C - C.shift(1)) * V                # one-day force: price move times volume
df["ForceIndex20"] = ema(df["ForceIndex1"], 20)          # smoothed 20-day version


# ---------------------------------------------------------------------------
# 7. Lagged prices (what the model can legitimately see from the recent past)
# ---------------------------------------------------------------------------
for k in (1, 2, 3, 5):
    df[f"Close_lag{k}"] = C.shift(k)                      # the close k days ago
df["Open_lag1"] = df["Open"].shift(1)                    # yesterday's open

# lags of the indices too, matching the provided sheet's idea
for k in (1, 2, 5):
    df[f"QQQ_lag{k}"] = df["QQQ_Close"].shift(k)
for k in (1, 5):
    df[f"SnP_lag{k}"]  = df["SnP_Close"].shift(k)
    df[f"DJIA_lag{k}"] = df["DJIA_Close"].shift(k)
for n in (10, 20, 50):
    df[f"QQQ_MA{n}"] = sma(df["QQQ_Close"], n)


# ---------------------------------------------------------------------------
# 8. Calendar features (pulled apart from the date)
# ---------------------------------------------------------------------------
d = df["Date"].dt
df["Day"] = d.day
df["DayofWeek"] = d.dayofweek                             # Monday = 0
df["DayofYear"] = d.dayofyear
df["Week"] = d.isocalendar().week.astype(int)
df["Month"] = d.month
df["Year"] = d.year
df["Is_month_end"]    = d.is_month_end.astype(int)        # 1 = yes, 0 = no
df["Is_month_start"]  = d.is_month_start.astype(int)
df["Is_quarter_end"]  = d.is_quarter_end.astype(int)
df["Is_quarter_start"]= d.is_quarter_start.astype(int)
df["Is_year_end"]     = d.is_year_end.astype(int)
df["Is_year_start"]   = d.is_year_start.astype(int)
df["Is_leap_year"]    = d.is_leap_year.astype(int)


# ---------------------------------------------------------------------------
# 9. Returns and the two targets
# ---------------------------------------------------------------------------
df["Return_1d"] = C.pct_change()                          # today's percentage change, a handy feature

# Option A target: the raw close price HORIZON days ahead.
df["target_price_5d"]  = C.shift(-HORIZON)
# Option B target: the percentage RETURN over the next HORIZON days (does not trend).
df["target_return_5d"] = C.shift(-HORIZON) / C - 1


# ---------------------------------------------------------------------------
# 10. Save to Excel
# ---------------------------------------------------------------------------
# The last HORIZON rows have no known future, so their targets are blank. That is
# correct and expected; we keep the rows so the feature history stays complete.
df.to_excel(OUT, index=False, sheet_name="AAPL_features")
print("saved:", OUT)
print("shape:", df.shape, " (rows, columns)")
print("date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("columns:", len(df.columns))
print("blank target rows at the end (expected):", int(df["target_price_5d"].isna().sum()))
