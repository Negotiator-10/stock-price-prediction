# Power BI Build Guide: AAPL Model Monitoring and Controls Dashboard

This dashboard presents the project the way an operations or model-risk team would monitor a
model: data controls, an exception log, a validation run register, and the performance detail
behind them. The Python code does all the modelling and testing. Power BI is the reporting and
controls layer on top. You build the .pbix yourself by following the steps below.

Framing to use with faculty: "The model is built and validated in Python. This Power BI report is
the operations and reporting view over it: data-quality controls, exception tracking, a run
register, and performance monitoring." Label it as a monitoring view over a one-off research run,
not a live production system. That reads as honest and mature.

Do NOT add a Power BI Forecast line to the price chart. The project's finding is that price and
direction are not predictable, so a forecast line would contradict the report.

---

## 1. The data you import

All tables are in this PowerBI folder. Import all 11 CSVs (Home > Get data > Text/CSV):

Operations and controls tables:

| Table | What it holds | Page |
|---|---|---|
| pbi_data_controls.csv | Data-quality and leakage controls with Pass / Info status | 1 |
| pbi_exception_log.csv | Control breaches found in validation, with severity and resolution | 1 |
| pbi_run_register.csv | Every validation run, its purpose, result and verdict | 2 |
| pbi_kpi.csv | 6 headline numbers | 1 |

Performance detail tables (the evidence behind the controls):

| Table | What it holds | Page |
|---|---|---|
| pbi_model_comparison.csv | Every model, both targets, RMSE, Dir accuracy, beats-naive flag | 3 |
| pbi_walkforward_rmse.csv | RMSE per fold, model vs naive (long format) | 3 |
| pbi_walkforward_summary.csv | Per-fold directional accuracy and beats-naive | 3 |
| pbi_volatility.csv | Volatility predictors (baselines vs models) and R2 | 4 |
| pbi_backtest_costs.csv | Strategy return and Sharpe at different trading costs | 4 |
| pbi_backtest_thresholds.csv | Strategy behaviour as the signal threshold rises | 4 |
| pbi_price_history.csv | AAPL daily Close and Volume, 2005 to 2020 (context only) | 4 |

No relationships are required. Each visual reads from one table. If Power BI auto-creates
relationships, delete them in Model view to keep it clean.

---

## 2. Load and check types (Power Query)

Transform data, then confirm:

- pbi_price_history Date set to Date.
- RMSE, MAE, R2, Sharpe, Pct, Value, Breach columns set to Decimal Number.
- Fold, Trades set to Whole Number.
- All label columns (Control, Status, Severity, Model, Target, Verdict, etc.) set to Text.

Close & Apply.

Optional but recommended: add a Sort helper for Severity so exceptions sort by importance.
In pbi_exception_log, add a custom column Severity Rank with values Critical=1, High=2, Medium=3,
Low=4, then set Severity to Sort by Severity Rank (Column tools > Sort by column).

---

## 3. Page 1: Controls and exceptions (the operations front page)

Goal: show at a glance that the data is clean and every control breach was caught and resolved.

1. Six Card visuals from pbi_kpi.csv. Drag Value into each card and filter the card to one
   Metric (Visual filter > Metric). Label each with a text box.

2. Data controls table from pbi_data_controls.csv: Control, Description, Result, Status.
   Conditional-format Status (Format > Cell elements > Background color): Pass green, Info grey,
   Fail red. This is the data-quality control panel.

3. Exception log table from pbi_exception_log.csv: Exception ID, Control, Observation, Severity,
   Status, Resolution. Conditional-format Severity (Critical dark red, High red, Medium amber,
   Low grey) and Status (Closed green). Sort by Severity.

4. One text box headline, for example:
   "All 7 control breaches identified and resolved. The largest, a Sharpe above 3 in backtest,
   traced to look-ahead leakage and was fixed by purging."

Reading of this page: a reader sees clean data controls and a fully worked exception log. This is
the operations and reporting story.

---

## 4. Page 2: Validation run register (the model-risk log)

Uses pbi_run_register.csv.

1. Table: Run ID, Analysis, Purpose, Key Result, Verdict, Status.
   Conditional-format Verdict: green for "Predictable (positive finding)", red for "Edge rejected"
   / "No edge" / "Direction not real", amber for "Edge on single window" / "Edge not robust".

2. Optional donut or bar: count of runs by Verdict, to show how many investigations were run
   (ten) and how they resolved.

3. Text box: "Ten validation runs were logged. Direction and price prediction were rejected after
   controls. Volatility prediction is the one accepted positive result."

This page reads like a model-validation inventory, which is exactly model-risk and reporting work.

---

## 5. Page 3: Performance detail (why price prediction was rejected)

Uses pbi_model_comparison.csv with a Slicer on Target (Price vs Return), plus the walk-forward.

1. Clustered bar: Axis = Model, Value = RMSE, Legend = Beats Naive, sorted by RMSE.
2. Clustered bar: Axis = Model, Value = Dir Accuracy Pct, with a constant line at 50 (Analytics
   pane) labelled "coin flip".
3. Clustered column from pbi_walkforward_rmse.csv: Axis = Fold, Value = RMSE, Legend = Model.
   Model and naive sit almost on top of each other every fold.
4. Table from pbi_walkforward_summary.csv with Beats Naive conditional-formatted. Yes in only
   3 of 6 folds.

---

## 6. Page 4: Volatility and backtest (the real edge, and the leak)

1. Bar from pbi_volatility.csv: Axis = Predictor, Value = R2, Legend = Type. The model bar (0.42)
   clears the best baseline (0.22). This is the accepted positive finding.
2. Table from pbi_backtest_costs.csv: Cost Bps Per Trade, Total Return Pct, Sharpe, Trades.
   Text box: "Trades = 1 and Pct Time Invested = 100, so the raw return was buy and hold, not a
   trading edge."
3. Optional line from pbi_price_history.csv (Date, Close) titled "AAPL historical price". Context
   only, no forecast line.

---

## 7. Formatting tips (keep it clean and human)

- One accent colour plus grey. Grey for baselines and Info, the accent only where the story is.
- Consistent status colours everywhere: green Pass / Closed / positive, red breach / rejected,
  amber caution.
- Round on display: RMSE 1 decimal, R2 2 decimals, percentages 0 to 1 decimal.
- Plain visual titles that state the takeaway, for example "All data controls passed" and
  "No model beats the naive baseline on price".
- Four tight pages beat one crowded page.
