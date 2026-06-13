"""
One-off backfill script: fetches missing delta from yfinance and pushes to CockroachDB.
Run this after the bug fix to sync the DB with any days missed since last seed.
"""
import os
import sys
import pandas as pd
import pandas_ta as ta
sys.path.insert(0, r'c:\userdata\repo\NSEStockDataAnalysis')
import db_utils
import fetch_data
from add_metrics import calculate_angle

DATA_FILE = r'c:\userdata\repo\NSEStockDataAnalysis\nse_stock_data_with_metrics_v2.csv'
RAW_FILE  = r'c:\userdata\repo\NSEStockDataAnalysis\nse_stock_data_backfill.csv'

# Step 1: Get max date from DB
db_max_date = db_utils.get_max_date()
print(f"DB max date: {db_max_date}")

# Step 2: Get max date from local CSV
csv_df = pd.read_csv(DATA_FILE, usecols=['Date'], parse_dates=['Date'])
csv_max_date = csv_df['Date'].max()
print(f"CSV max date: {csv_max_date.date()}")

# Step 3: Fetch delta from yfinance
if db_max_date is not None:
    delta_start = db_max_date + pd.Timedelta(days=1)
else:
    delta_start = csv_max_date + pd.Timedelta(days=1)

print(f"Fetching delta from yfinance starting {delta_start.date()}...")
fetch_data.fetch_stock_data(RAW_FILE, start_date=delta_start)

if not os.path.exists(RAW_FILE):
    print("No new data fetched from yfinance. DB may already be up to date.")
    sys.exit(0)

new_df = pd.read_csv(RAW_FILE)
new_df['Date'] = pd.to_datetime(new_df['Date'])
print(f"Fetched {len(new_df)} new rows from yfinance. Date range: {new_df['Date'].min().date()} to {new_df['Date'].max().date()}")

if new_df.empty:
    print("No new rows. Nothing to do.")
    sys.exit(0)

# Step 4: Load CSV base + append new rows to compute metrics
print("Loading base data from CSV...")
base_df = pd.read_csv(DATA_FILE)
base_df['Date'] = pd.to_datetime(base_df['Date'])

# Remove overlap
base_df = base_df[~base_df['Date'].isin(new_df['Date'].unique())]
raw_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
merged_df = pd.concat([
    base_df[[c for c in raw_cols if c in base_df.columns]],
    new_df[[c for c in raw_cols if c in new_df.columns]]
], ignore_index=True)

# Step 5: Recompute metrics on full merged dataset
print("Computing metrics...")
merged_df.sort_values(by=['Ticker', 'Date'], inplace=True)

merged_df['RSI_14'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.rsi(x, length=14))
for w in [3, 20, 100, 200]:
    merged_df[f'{w}DMA'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=w))
for w in [3, 20, 100, 200]:
    def vwma(g, ww=w):
        r = ta.vwma(g['Close'], g['Volume'], length=ww)
        if r is not None:
            return r
        return (g['Close'] * g['Volume']).rolling(ww, min_periods=1).sum() / g['Volume'].rolling(ww, min_periods=1).sum()
    merged_df[f'{w}VWMA'] = merged_df.groupby('Ticker', group_keys=False).apply(vwma)

merged_df['HA_Close'] = (merged_df['Open'] + merged_df['High'] + merged_df['Low'] + merged_df['Close']) / 4
init_ha = (merged_df['Open'] + merged_df['Close']) / 2
x = merged_df.groupby('Ticker')['HA_Close'].shift(1).fillna(init_ha)
merged_df['HA_Open'] = x.groupby(merged_df['Ticker']).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
for col in [f'{w}DMA' for w in [3, 20, 100, 200]] + ['RSI_14']:
    merged_df[f'{col}_SLOPE'] = merged_df.groupby('Ticker')[col].transform(lambda x: calculate_angle(x, 3))

# Step 6: Extract only the delta rows (after DB max date) and upsert
delta_df = merged_df[merged_df['Date'] > db_max_date] if db_max_date else merged_df
print(f"Upserting {len(delta_df)} delta rows to CockroachDB...")
count = db_utils.upsert_metrics(delta_df)
print(f"Done! Upserted {count} rows.")

# Step 7: Save updated CSV
print("Saving updated CSV...")
merged_df.sort_values(by=['Ticker', 'Date'], ascending=[True, False], inplace=True)
merged_df.to_csv(DATA_FILE, index=False)
print("CSV saved.")

# Verify
new_db_max = db_utils.get_max_date()
print(f"DB max date after backfill: {new_db_max.date() if new_db_max else 'N/A'}")
