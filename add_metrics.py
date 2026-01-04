import pandas as pd
import numpy as np
import os
import time
import fetch_data
import warnings

warnings.filterwarnings('ignore')

REQUIRED_FILE = "nse_stock_data.csv"
OUTPUT_FILE = "nse_stock_data_with_metrics.csv"
FRESHNESS_HOURS = 8

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_angle(series, window=1):
    # Angle calculation: degrees(arctan(pct_change * 100))
    # This treats a 1% change over the window as a 45-degree angle.
    # User formula: np.degrees(np.atan(data_series.pct_change(periods=window) * 100))
    #change = series.pct_change(periods=window) * 100
    #angle = np.degrees(np.arctan(change))
    angle = np.degrees(np.atan(series.pct_change(periods=window) * 100))
    return angle


def main():
    # 1. Freshness Check
    should_fetch = False
    if not os.path.exists(REQUIRED_FILE):
        print(f"{REQUIRED_FILE} not found. Fetching data...")
        should_fetch = True
    else:
        file_mod_time = os.path.getmtime(REQUIRED_FILE)
        current_time = time.time()
        hours_diff = (current_time - file_mod_time) / 3600
        
        if hours_diff > FRESHNESS_HOURS:
            print(f"Data is older than {FRESHNESS_HOURS} hours ({hours_diff:.2f} hrs). Fetching new data...")
            should_fetch = True
        else:
            print(f"Data is fresh ({hours_diff:.2f} hrs old). Loading local file...")

    if should_fetch:
        fetch_data.fetch_stock_data(REQUIRED_FILE)

    # 2. Load Data
    try:
        df = pd.read_csv(REQUIRED_FILE)
    except Exception as e:
        print(f"Error reading file {REQUIRED_FILE}: {e}")
        return

    # Ensure Date is sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    print("Calculating metrics...")

    # 3. Calculate Metrics (Group by Ticker to treat each stock independently)
    # We use transform when possible for index safety and apply for multi-column.
    # We apply shift(-1) to "move up" the data as requested by the user.
    
    # RSI
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))
    
    # DMAs and EMAs
    for window in [3, 20, 100, 200]:
        df[f'{window}DMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{window}EMA'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=window, adjust=False, min_periods=1).mean())
        
    # VWMAs (Requires multi-column, so we use apply + explicit alignment)
    for window in [3, 20, 100, 200]:
        name = f'{window}VWMA'
        def compute_group_vwma(g):
            vwma = (g['Close'] * g['Volume']).rolling(window, min_periods=1).sum() / g['Volume'].rolling(window, min_periods=1).sum()
            return vwma
        
        # Explicitly aligning the result back to the dataframe
        df[name] = df.groupby('Ticker', group_keys=False).apply(compute_group_vwma)

    # 4. Calculate Angles for all metrics
    metrics_cols = [f'{w}DMA' for w in [3, 20, 100, 200]] + \
                   [f'{w}EMA' for w in [3, 20, 100, 200]] + \
                   [f'{w}VWMA' for w in [3, 20, 100, 200]] + \
                   ['RSI_14']
    
    for col in metrics_cols:
        angle_col = f'{col}_LINE_ANGLE' if 'RSI' not in col else f'{col}_angle'
        # We also shift the angle up by 1 row to fix the alignment lag
        df[angle_col] = df.groupby('Ticker')[col].transform(lambda x: calculate_angle(x,3))

    print(f"Saving to {OUTPUT_FILE}...")
    # Final Sort: Ticker Ascending, Date Descending (Latest date on top)
    df.sort_values(by=['Ticker', 'Date'], ascending=[True, False], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
