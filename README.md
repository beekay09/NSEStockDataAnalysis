# NSE Stock Data Analysis & Technical Metrics

This project fetches historical stock data for over 200 NSE (National Stock Exchange of India) listed companies and calculates a comprehensive suite of technical indicators.

## Features

- **Automated Data Fetching**: Retrieves 730 days (2 years) of OHLCV data using `yfinance`.
- **Data Freshness Check**: Implements an 8-hour cache logic. New data is only fetched if the local `nse_stock_data.csv` is missing or older than 8 hours.
- **Robust Technical Metrics**: Calculates metrics per ticker with index safety using pandas `transform` and `groupby`.
- **Unified Output**: Generates a consolidated CSV with all raw data and calculated metrics.

## Technical Metrics Included

For each stock, the following metrics are calculated:

| Metric | Windows / Periods | Description |
| :--- | :--- | :--- |
| **DMA** (Daily Moving Average) | 3, 20, 100, 200 | Simple moving average of the Closing price. |
| **EMA** (Exponential Moving Average) | 3, 20, 100, 200 | Exponentially weighted moving average. |
| **VWMA** (Volume Weighted MA) | 3, 20, 100, 200 | Moving average weighted by trading volume. |
| **RSI** (Relative Strength Index) | 14 | Momentum oscillator for overbought/oversold conditions. |
| **Line Angles** | All above | Calculated as `degrees(arctan(pct_change * 100))` to quantify trend steepness. |

## Environment Setup

### 1. Python Configuration
The project is optimized for **Python 3.13.9**.
- **Installation Path**: `C:\Python313`
- **Scripts Path**: `C:\Python313\Scripts`

### 2. Virtual Environment
It is recommended to use the provided virtual environment:
```powershell
# Activate the environment (Windows)
.\.venv\Scripts\activate
```

### 3. Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

## Configuration

### Stock List
The list of stocks to analyze is maintained in `stocks.txt`.
- Add or remove stock tickers (e.g., `RELIANCE`, `TCS`) in this file, one per line.
- The scripts automatically append `.NS` for NSE compatibility.

## Usage

### Complete Workflow
The `add_metrics.py` script handles the entire pipeline (freshness check -> fetch -> calculate):
```bash
python add_metrics.py
```

### Manual Data Fetch (Optional)
To fetch raw data without calculating metrics:
```bash
python fetch_data.py
```

## Project Structure

- `fetch_data.py`: Core logic for interacting with the Yahoo Finance API. Reads tickers from `stocks.txt`.
- `add_metrics.py`: Main entry point for data validation and metrics calculation.
- `update_notebook.py`: Utility to inject validation logic into the analysis notebook.
- `stocks.txt`: Configuration file for the list of stocks to process.
- `requirements.txt`: List of Python dependencies (pandas, numpy, yfinance, etc.).
- `nse_stock_data.csv`: Raw historical data (Generated).
- `nse_stock_data_with_metrics.csv`: Final dataset including technical indicators (Generated).

## Technical Implementation Notes

- **Pandas Optimization**: Uses `groupby().transform()` for SMA/EMA/RSI to handle multiple stocks in a single dataframe efficiently.
- **VWMA Calculation**: Uses `groupby().apply()` with volume-weighted summation to ensure accuracy across tickers.
- **Error Handling**: Includes basic exception handling for file I/O and network requests.
