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
| **VWMA** (Volume Weighted MA) | 3, 20, 100, 200 | Moving average weighted by trading volume. |
| **RSI** (Relative Strength Index) | 14 | Momentum oscillator for overbought/oversold conditions. |
| **Slopes** (Trend Angles) | DMAs & RSI | Calculated as `degrees(atan(pct_change * 100))` to quantify trend steepness. Suffix: `_SLOPE` |

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

## 📊 Streamlit Dashboard

Analyze stocks interactively using the built-in dashboard.

### Running the Dashboard
```bash
streamlit run App.py
```

### Key Features
- **10+ Analysis Strategies**: Dedicated tabs for Low RSI, Low Beta, Breakouts, Crossovers, and more.
- **Interactive Charts**: Zoomable Plotly charts with Heikin Ashi candles, Bollinger Bands, and Volume Profile.
- **Sidebar Filters**: Dynamic filters for each strategy (Timeframe, RSI Thresholds, Moving Average angles).
- **Data Export**: Copy data to clipboard or download as CSV.

## Project Structure

- `App.py`: Main Streamlit application file (UI Layer).
- `analysis_logic.py`: Core logic for stock filtering and strategy implementation.
- `app_documentation.json`: Configurable text for strategies and help.
- `style.css`: Custom CSS for the dashboard.
- `fetch_data.py`: Core logic for interacting with the Yahoo Finance API. Reads tickers from `stocks.txt`.
- `add_metrics.py`: Main entry point for data validation and metrics calculation.
- `update_notebook.py`: Utility to inject validation logic into the analysis notebook.
- `stocks.txt`: Configuration file for the list of stocks to process.
- `requirements.txt`: List of Python dependencies (pandas, numpy, yfinance, etc.).
- `nse_stock_data.csv`: Raw historical data (Generated).
- `nse_stock_data_with_metrics.csv`: Final dataset including technical indicators (Generated).

## Technical Implementation Notes

- **Modular Architecture**:
    - **UI**: `App.py` handles the layout and user interaction.
    - **Logic**: `analysis_logic.py` contains the business logic and filtering algorithms.
    - **Config**: Documentation and Styling are separated into JSON and CSS files.
- **Pandas Optimization**: Uses `groupby().transform()` for SMA/RSI to handle multiple stocks in a single dataframe efficiently.
- **VWMA Calculation**: Uses `groupby().apply()` with volume-weighted summation to ensure accuracy across tickers.
- **Error Handling**: Includes basic exception handling for file I/O and network requests.
