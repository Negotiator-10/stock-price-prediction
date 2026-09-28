# AAPL Stock Price Prediction and Model Monitoring

Can a model forecast Apple's stock one week ahead, and how do you prove a good-looking result is real?
This project tests that question end to end and presents the findings as a model-risk and controls
dashboard in Power BI.

**Short answer:** price direction was not predictable once every source of look-ahead was removed, but
volatility was (R squared 0.42 against 0.22 for a persistence baseline).

## Data

3,732 daily rows of AAPL (October 2005 to August 2020), 64 columns: prices, volume, index closes and
technical indicators. The dataset was provided by the IIQF course and is not included here; place it at
`data/AAPL.csv` to rerun the scripts. `build_features.py` recomputes every indicator from the raw prices
to check the provided columns.

## What was done

1. **Baseline models on price.** 8 models from linear regression to LSTM, trained on 80% of the history
   and tested on the rest. None beat the naive forecast (next week's price = today's): best RMSE 11.62
   against 10.40 for naive. Tree models and the LSTM failed worst because the test period sits above any
   price they had seen, and trees cannot extrapolate.
2. **Returns instead of price.** Forecasting 5-day returns fixed the extrapolation problem, but the
   walk-forward hit rate (about 58%) equalled the share of up weeks: no skill beyond market drift.
3. **Leakage post-mortem.** A rolling-window model appeared to show a strong momentum edge, and a
   cost-adjusted backtest returned a Sharpe ratio of 3.56. That was too good to be true: the training
   window included labels that were only known after the forecast date. Purging those rows made the
   edge disappear at every window length, on fresh stocks (MSFT, JPM, KO) and in every volatility regime.
4. **Honest testing.** Direction classifier AUC 0.508 (confidence interval includes 0.5), no edge at any
   horizon from 1 to 60 days, and a multiple-testing correction across about 44 trials.
5. **What does work.** Next-week volatility is predictable from current volatility measures
   (R squared 0.42, correlation 0.69), which is consistent with GARCH-style volatility clustering.

## Power BI dashboard

`PowerBI/AAPL_Model_Monitoring.pbix`, 4 pages:

| Page | Shows |
|---|---|
| Controls and exceptions | Data quality checks and the exception log |
| Validation run register | Every experiment, its result and verdict |
| Performance detail | Why price prediction was rejected |
| Volatility and backtest | The real finding, and the leak that looked like one |

## Files

| Path | Contents |
|---|---|
| `stock_forecast.py` | Main pipeline: features, 8 models, price and return modes |
| `walk_forward.py`, `purged_main.py`, `purged_robustness.py` | Walk-forward and purged (leak-free) testing |
| `rolling_*.py`, `shortwin_backtest.py`, `backtest.py` | The rolling-window experiments and the backtest that exposed the leak |
| `edge_check.py`, `horizon_sweep.py`, `classifier_direction.py`, `multiple_testing.py` | Significance and robustness tests |
| `test2_freshstock.py`, `test3_regime.py` | Out-of-sample checks on other stocks and market regimes |
| `vol_predict.py` | Volatility forecasting |
| `outputs/` | Charts and result tables |
| `PowerBI/` | Dashboard and the CSVs it loads |
| `Stock_Price_Prediction_Report.docx` | Full written report |

**Tools:** Python (pandas, scikit-learn, XGBoost, TensorFlow), Power BI.

**Key lesson:** out-of-sample testing catches overfitting but not look-ahead leakage. A result that looks
too good is treated as a bug until explained.
