# Gold Price Forecasting with ARIMA and GARCH

Time series forecasting of SHFE gold futures (AU_SHF) daily closing prices using ARIMA(0,0,1) and GARCH(1,1) with Student-t errors. Evaluated via expanding-window rolling forecasts at 1-day and 20-day horizons.

## Steps to Reproduce

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook:
   ```bash
   jupyter notebook code/Gold_Price_Forecasting.ipynb
   ```
   Graphs will be saved to `graphs/`.

3. Read the write-up in `blog/gold_price_forecast_blog.md`.

## Folder Structure

```
├── blog/
│   └── gold_price_forecast_blog.md
├── code/
│   └── Gold_Price_Forecasting.ipynb
├── data/
│   └── AU_SHF_EN.csv
├── graphs/
│   ├── eda_gold_closing_price.png
│   ├── eda_log_returns.png
│   ├── weekly_seasonal_decomposition.png
│   ├── monthly_seasonal_decomposition.png
│   ├── acf_pacf.png
│   ├── forecast_split.png
│   ├── rolling_forecasts_h1.png
│   └── rolling_forecasts_h20.png
├── requirements.txt
└── README.md
```
