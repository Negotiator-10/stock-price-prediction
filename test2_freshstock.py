"""
TEST 2 of 3: does the edge appear on FRESH stocks? (leaky vs purged)
===================================================================
Download stocks we have never touched (MSFT, JPM, KO), build a price feature set,
and run the short-window directional test both leaky and purged. If the "edge" is
a real signal it should show up somewhere; if it is the look-ahead leak, we expect
the SAME pattern as Apple: a big leaky edge that vanishes after purging.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

HORIZON, STEP, WINDOW, START = 5, 5, 63, 300

def features(d):
    C, H, L, V = d["Close"], d["High"], d["Low"], d["Volume"]
    f = pd.DataFrame(index=d.index)
    for k in [1, 2, 3, 5]: f[f"ret{k}"] = C.pct_change(k)
    for n in [5, 10, 20, 50]: f[f"maR{n}"] = C / C.rolling(n).mean()
    for n in [5, 10, 20]: f[f"roc{n}"] = C.pct_change(n)
    for n in [5, 20]: f[f"vol{n}"] = C.pct_change().rolling(n).std()
    f["hl"] = (H - L) / C
    f["volR"] = V / V.rolling(20).mean()
    delta = C.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + rs)
    f["ret_fwd5"] = C.shift(-HORIZON) / C - 1          # target
    return f.dropna()

def test(f, purge):
    X = f.drop(columns=["ret_fwd5"]).values
    y = f["ret_fwd5"].values
    tu, mu = [], []
    for t in range(START, len(f) - HORIZON, STEP):
        end = t - HORIZON if purge else t
        lo = max(0, end - WINDOW)
        if end - lo < 10: continue
        sc = StandardScaler().fit(X[lo:end])
        m = Lasso(alpha=0.001, max_iter=5000).fit(sc.transform(X[lo:end]), y[lo:end])
        pr = m.predict(sc.transform(X[t:t+1]))[0]
        tu.append(int(y[t] > 0)); mu.append(int(pr > 0))
    tu = np.array(tu); mu = np.array(mu); n = len(tu)
    correct = int((tu == mu).sum()); acc = correct/n*100; base = tu.mean()*100
    p = binomtest(correct, n, tu.mean(), alternative="greater").pvalue
    return round(acc-base, 1), ("YES" if p < 0.05 else "no")

print("=" * 74)
print("TEST 2: FRESH STOCKS, edge over always-up (leaky vs purged)")
print("=" * 74)
rows = []
for tk in ["MSFT", "JPM", "KO"]:
    d = yf.download(tk, start="2007-01-01", end="2020-08-13", progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    f = features(d)
    le, ls = test(f, purge=False)
    pe, ps = test(f, purge=True)
    rows.append({"stock": tk, "n_days": len(f), "leaky_edge": le, "leaky_sig": ls,
                 "PURGED_edge": pe, "PURGED_sig": ps})
    print(f"  {tk} done")
print()
print(pd.DataFrame(rows).to_string(index=False))
print("\nIf leaky shows an edge everywhere but purged does not, the 'edge' is the")
print("look-ahead leak, confirmed on stocks we never studied. Not Apple-specific.")
