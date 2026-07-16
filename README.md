# Forecasting Weekly German Electricity Demand

An end-to-end time-series study that forecasts the weekly average electricity load of the
German transmission system and asks a single sharp question: **over a two-year horizon, can
any statistical or machine-learning model genuinely beat the simplest seasonal benchmark —
and where the numbers say yes, is the gain real or an artefact of how the forecast is made?**

## Overview

The hourly German load series (Open Power System Data, Jan 2015 – Oct 2020) is aggregated to a
weekly grid of 301 observations. The final two years (104 weeks) are held out as a **genuine
multi-step forecast horizon** — every model is scored on exactly the same test window and, where
applicable, forecasts once at the origin using no test-period data.

Five model families are compared:

| Family | Models |
|---|---|
| Benchmarks | Mean, Naive, Seasonal naive, Drift |
| Statistical | SARIMA `(0,1,2)(0,1,1,52)` (AIC grid search) |
| Statistical + exogenous | SARIMAX with Berlin temperature / heating- & cooling-degree terms |
| Feature-based | Random Forest & Gradient Boosting (recursive **and** 1-step) |
| Neural | LSTM on the hourly signal, aggregated to weekly |

All models are evaluated with **MAE, RMSE, MASE and bias**.

## Key finding

Compared **like-for-like as true multi-step forecasts, no model meaningfully beats the
seasonal-naive benchmark** (MASE ≈ 1.73) over this horizon. The models that appear to win —
the walk-forward LSTM and the one-step tree ensembles — do so only because they are fed recent
*actual* load (a form of multi-step forecast leakage), not because of genuine long-horizon skill.
The elevated errors across every method trace to the **2020 COVID-19 demand collapse**, which
falls inside the test window and is unforecastable from pre-2019 data.

## Notebook structure

The analysis lives in a single, top-to-bottom runnable notebook (`Run all`). All data is
downloaded inside the notebook — no manual uploads.

- **Part 1** — Data acquisition, EDA, seasonality (STL) and stationarity (ADF, KPSS, differencing, ACF/PACF)
- **Part 2** — Benchmark forecasts (mean, naive, seasonal naive, drift)
- **Part 3** — SARIMA: AIC grid search, residual diagnostics (Ljung–Box), forecast + confidence intervals
- **Part 4** — SARIMAX with temperature (conditional / explanatory forecast)
- **Part 5** — Feature-based regression (Random Forest & Gradient Boosting; recursive vs 1-step)
- **Part 6** — LSTM on hourly data
- **Part 7** — Analysis answers (model comparison, leakage, differencing justification, covariates, interpretability, operational recommendation)
- **Part 8** — Final comparison table + figure

## Data sources

- **Load:** Open Power System Data — `time_series_60min_singleindex` (2020-10-06 release)
- **Weather:** Open-Meteo Historical Weather API (Berlin daily mean temperature)

## Running it

Colab-ready. For a stronger LSTM, switch to a GPU runtime and raise the settings noted in the
Part 6 config cell (longer look-back, deeper network, more epochs, no training subsample). To run
the full required SARIMA `p,d,q` grid search, set `FULL_SEARCH = True` in Part 3 (~20–30 min).

```bash
pip install statsmodels scikit-learn tensorflow pandas numpy matplotlib requests
jupyter notebook German_Load_Forecasting.ipynb
```

## Results (2-year weekly forecast, sorted by MASE)

| Model | MAE | RMSE | MASE | Bias | Forecast type |
|---|---|---|---|---|---|
| LSTM (hourly→weekly) | 1.176 | 1.623 | 0.879 | -1.050 | 1-step (leak) |
| Random Forest (1-step) | 1.927 | 2.600 | 1.440 | 1.135 | 1-step (leak) |
| Gradient Boosting (1-step) | 1.999 | 2.704 | 1.494 | 1.287 | 1-step (leak) |
| **Seasonal naive** | **2.319** | **3.007** | **1.732** | **1.732** | **Multi-step** |
| Random Forest (recursive) | 2.383 | 3.018 | 1.781 | 1.274 | Multi-step |
| Gradient Boosting (recursive) | 2.531 | 3.317 | 1.891 | 1.561 | Multi-step |
| SARIMAX (+temp, conditional) | 2.851 | 3.589 | 2.131 | 2.557 | Multi-step* |
| SARIMA | 3.126 | 3.826 | 2.336 | 2.872 | Multi-step |
| Naive | 3.783 | 4.459 | 2.827 | -0.882 | Multi-step |
| Mean | 3.789 | 4.397 | 2.831 | 0.481 | Multi-step |
| Drift | 4.340 | 5.118 | 3.243 | 1.007 | Multi-step |

Seasonal naive is the best **genuine** multi-step forecast; rows above it consume recent actual
load, and the SARIMAX row is conditioned on observed future temperature.

## Possible improvements

- Evaluate SARIMAX with **forecast** rather than observed temperature for an honest operational error
- Rolling-origin (backtesting) evaluation over shorter horizons
- Deterministic calendar / public-holiday regressors
- Intervention / anomaly handling for structural breaks (e.g. a lockdown indicator)
- A direct multi-step (sequence-to-sequence) neural forecaster evaluated on the same origin-fixed basis

## Tech stack

Python · pandas · NumPy · statsmodels · scikit-learn · TensorFlow/Keras · Matplotlib
