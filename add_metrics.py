import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import time
import datetime
import fetch_data
import warnings

try:
    import db_utils
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

warnings.filterwarnings('ignore')

REQUIRED_FILE = "nse_stock_data.csv"
OUTPUT_FILE = "nse_stock_data_with_metrics_v2.csv"
FRESHNESS_HOURS = 8

def calculate_angle(series, window=1):
    # Angle calculation: degrees(arctan(pct_change * 100))
    # This treats a 1% change over the window as a 45-degree angle.
    # User formula: np.degrees(np.atan(data_series.pct_change(periods=window) * 100))
    angle = np.degrees(np.atan(series.pct_change(periods=window) * 100))
    return angle


def main():
    # 0. Output Freshness Check
    # 0. Output Freshness Check
    if os.path.exists(OUTPUT_FILE):
        output_mod_time = os.path.getmtime(OUTPUT_FILE)
        
        # Smart Refresh Logic
        current_dt = datetime.datetime.now()
        file_mod_dt = datetime.datetime.fromtimestamp(output_mod_time)
        
        # Smart Refresh Logic
        current_dt = datetime.datetime.now()
        file_mod_dt = datetime.datetime.fromtimestamp(output_mod_time)
        
        # Check if it is "Post Market" (After 3:31 PM)
        is_post_market = (current_dt.hour > 15) or (current_dt.hour == 15 and current_dt.minute >= 31)
        
        # Check if file was updated BEFORE 3:31 PM today
        # Construct "Today 3:31 PM"
        today_cutoff = current_dt.replace(hour=15, minute=31, second=0, microsecond=0)
        
        should_smart_update = is_post_market and (file_mod_dt < today_cutoff)
        
        if should_smart_update:
             print(f"It is after 3:31 PM and data is from {file_mod_dt}. Updating for latest market data...")
             # Pass through to fetch data
        elif (time.time() - output_mod_time) / 3600 < FRESHNESS_HOURS:
            print(f"Output file {OUTPUT_FILE} is fresh. Skipping update.")
            return

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
    
    print("Calculating metrics using pandas_ta...")

    # 3. Calculate Metrics (Group by Ticker to treat each stock independently)

    # RSI using pandas_ta
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.rsi(x, length=14))
    
    # DMAs (SMA) using pandas_ta
    for window in [3, 20, 100, 200]:
        df[f'{window}DMA'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=window))
        
    # VWMAs using pandas_ta
    for window in [3, 20, 100, 200]:
        name = f'{window}VWMA'
        def compute_group_vwma(g, w=window):
            result = ta.vwma(g['Close'], g['Volume'], length=w)
            if result is not None:
                return result
            # Fallback to manual calculation
            vwma = (g['Close'] * g['Volume']).rolling(w, min_periods=1).sum() / g['Volume'].rolling(w, min_periods=1).sum()
            return vwma
        
        # Explicitly aligning the result back to the dataframe
        df[name] = df.groupby('Ticker', group_keys=False).apply(compute_group_vwma)

    # 4. Calculate Heikin Ashi
    print("Calculating Heikin Ashi...")
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    initial_ha_open = (df['Open'] + df['Close']) / 2
    x = df.groupby('Ticker')['HA_Close'].shift(1).fillna(initial_ha_open)
    df['HA_Open'] = x.groupby(df['Ticker']).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())

    # 4. Calculate Angles for all metrics
    metrics_cols = [f'{w}DMA' for w in [3, 20, 100, 200]] + \
                   ['RSI_14']
                  # [f'{w}EMA' for w in [3, 20, 100, 200]] + \
                  # [f'{w}VWMA' for w in [3, 20, 100, 200]] + \
    
    for col in metrics_cols:
        angle_col = f'{col}_SLOPE' 
        # We also shift the angle up by 1 row to fix the alignment lag
        df[angle_col] = df.groupby('Ticker')[col].transform(lambda x: calculate_angle(x,3))

    print(f"Saving to {OUTPUT_FILE}...")
    # Final Sort: Ticker Ascending, Date Descending (Latest date on top)
    df.sort_values(by=['Ticker', 'Date'], ascending=[True, False], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("CSV saved.")

    # 5. Upsert only the delta to CockroachDB
    if DB_AVAILABLE:
        try:
            print("Syncing delta metrics to CockroachDB...")
            db_max_date = db_utils.get_max_date()
            if db_max_date is not None:
                delta_df = df[df['Date'] > db_max_date]
                print(f"DB has data up to {db_max_date.date()}. Upserting {len(delta_df)} new rows...")
            else:
                delta_df = df
                print(f"DB is empty. Upserting all {len(delta_df)} rows...")
            if not delta_df.empty:
                db_utils.upsert_metrics(delta_df)
            else:
                print("No new rows to upsert.")
        except Exception as e:
            print(f"Warning: DB sync failed (non-fatal): {e}")
    else:
        print("db_utils not available. Skipping DB sync.")

    print("Done.")

if __name__ == "__main__":
    main()

