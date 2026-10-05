# LSTM Stock Forecasting: GOOG, AAPL, MSFT (May 2026 Outlook)

Group assignment for **DA4131 - Advanced ML Applications for Business**, Department of Decision Sciences, Faculty of Business, University of Moratuwa (Semester 07).

The project trains one LSTM model per stock to predict the next-day price movement, evaluates the models on a held-out test period, and makes a blind forecast for **1 to 14 May 2026**.


## Main idea

The models predict the **daily log return**, not the raw price.
An ADF test in the notebook shows that prices are non-stationary (p > 0.5 for all three stocks) while returns are stationary (p < 0.001). A model trained on raw prices cannot predict above the highest price it saw in training, which fails when a stock reaches new highs.
The predicted returns are converted back to prices, starting from the real close of 30 April 2026.

## What the notebook does

| Task | Content |
|---|---|
| 1. Data | Five years of daily OHLCV data from Yahoo Finance (`yfinance`), 1 Apr 2021 to 31 Mar 2026. Saved as CSV files. |
| 2. EDA and preprocessing | Price and relative performance charts, candlestick with volume, return distribution, rolling volatility, moving averages, correlations, decomposition, ADF test, largest daily moves, April 2026 review. Log returns, standardisation (fitted on training data only), 60-day windows, 80/20 chronological split. |
| 3. Model | Two stacked LSTM layers with dropout, a Dense(32) layer and a one-unit output. Huber loss, Adam, early stopping. Grid search over units {32, 64, 100} and dropout {0.2, 0.3, 0.4}, chosen by validation loss. |
| 4. Evaluation | MAE, RMSE, MAPE, R2 on price and on returns, directional accuracy. Compared with the initial model and a "tomorrow = today" baseline. |
| 5. Forecast | 10 trading days (1 to 14 May 2026), recursive forecast with the saved tuned model. No retraining. Approximate 80% range. |

## Results

**Test period:** 1 Apr 2025 to 31 Mar 2026 (one-step-ahead predictions).

| Stock | Tuned MAE | Baseline MAE | Directional accuracy | Share of up days | R2 on returns |
|---|---|---|---|---|---|
| GOOG | $3.36 | $3.24 | 53.4% | 53.0% | -0.105 |
| AAPL | $2.77 | $2.77 | 52.6% | 52.6% | 0.002 |
| MSFT | $4.84 | $4.84 | 51.0% | 51.8% | -0.001 |

The LSTM does not beat the "tomorrow = today" baseline. R2 on price is about 0.98 to 0.99 for every model, including the baseline, because each prediction starts from the real previous close. Daily stock returns are close to random, so this result is expected.

The grid search showed that units and dropout matter very little (validation losses within about 1.5% for each stock). The main improvement came from modelling returns and from fitting the scaler on training data only.

**Blind forecast (made with data up to 30 April 2026):**

| Stock | Last close (30 Apr) | Forecast 14 May | Change | 80% range on 14 May |
|---|---|---|---|---|
| GOOG | $381.46 | $355.82 | -6.7% | $328.60 to $385.30 |
| AAPL | $270.87 | $273.52 | +1.0% | $252.58 to $296.20 |
| MSFT | $406.13 | $408.13 | +0.5% | $381.52 to $436.59 |

**Check against real prices (run after 14 May):**

| Stock | MAE | MAPE | Days inside 80% range |
|---|---|---|---|
| GOOG | $31.98 | 8.16% | 10% |
| AAPL | $16.67 | 5.72% | 30% |
| MSFT | $4.75 | 1.15% | 100% |

MSFT stayed close to the forecast. GOOG and AAPL rose well above it. The model reads only past returns, so it cannot anticipate earnings reactions (both companies reported around 29 to 30 April). The forecast should be seen as a weak short-term signal, not a basis for trading decisions.

## Repository files

```
.
|-- DA4131_Group12_LSTM_Assignment_final.ipynb   # main notebook (Tasks 1 to 5, with outputs)
|-- DA4131_Group12_LSTM_Assignment.pdf           # PDF version of the notebook
|-- Output.csv                                   # forecast, 1 to 14 May 2026
|-- Forecast_with_range.csv                      # forecast with approximate 80% range
|-- GOOG_data.csv, AAPL_data.csv, MSFT_data.csv  # raw price data (Apr 2021 to Mar 2026)
|-- requirements.txt
`-- README.md
```

## How to run

1. Install the packages: `pip install -r requirements.txt`
2. Open the notebook in Jupyter or Google Colab and run all cells from top to bottom.
3. The grid search trains 36 small models, so it takes some time. A GPU runtime helps.

Notes:
- Data is downloaded live from Yahoo Finance, so a new run can differ slightly from the saved results.
- Random seeds are fixed, but exact results can still change between machines.
- The notebook uses plotly charts. GitHub does not show them. Open the notebook in Colab, or paste the GitHub link into [nbviewer](https://nbviewer.org), or read the PDF.

## Limitations

- One feature only (past returns of one stock). No news, earnings dates or market data.
- The recursive forecast moves toward the average return, so the forecast lines are smooth.
- The 80% range is approximate and based on the one-day test error.
- Only three stocks and one test year.

## Data source

Yahoo Finance, accessed with the `yfinance` Python package.
