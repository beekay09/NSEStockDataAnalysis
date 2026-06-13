"""
Diagnostic + fix script:
1. Finds all dates where fewer than 222 stocks have data in the DB
2. Checks if those are NSE holidays (expected) or genuine data gaps
3. Fills in any genuine gaps by re-fetching from yfinance
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
TOTAL_STOCKS = 222
THRESHOLD = 200   # flag any date with fewer than this many stocks

# NSE holidays — these dates will naturally have fewer or no records (market closed)
NSE_HOLIDAYS_2024_2026 = {
    "2024-10-02", "2024-11-01", "2024-11-15", "2024-12-25",
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31",
    "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
    "2025-06-07", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-20", "2025-10-21", "2025-10-24", "2025-11-05",
    "2025-12-25",
    "2026-01-26", "2026-02-26", "2026-03-20", "2026-03-30",
    "2026-04-02", "2026-04-06", "2026-04-10", "2026-04-14",
    "2026-05-01", "2026-05-25",
}

print("=" * 60)
print("Step 1: Querying DB for dates with partial data...")
print("=" * 60)

engine = db_utils.get_engine()
df = pd.read_sql(
    'SELECT "Date", COUNT(DISTINCT "Ticker") as ticker_count '
    'FROM stock_metrics '
    'GROUP BY "Date" '
    'ORDER BY "Date"',
    engine
)
df['Date'] = pd.to_datetime(df['Date'])

# Find dates below threshold
gaps = df[df['ticker_count'] < THRESHOLD].copy()
gaps['date_str'] = gaps['Date'].dt.strftime('%Y-%m-%d')
gaps['is_holiday'] = gaps['date_str'].isin(NSE_HOLIDAYS_2024_2026)
gaps['is_weekend'] = gaps['Date'].dt.dayofweek >= 5

print(f"\nDates with fewer than {THRESHOLD} stocks:")
print(f"{'Date':<15} {'Count':>6} {'Holiday':>8} {'Weekend':>8}")
print("-" * 40)
for _, row in gaps.iterrows():
    print(f"{row['date_str']:<15} {row['ticker_count']:>6} {str(row['is_holiday']):>8} {str(row['is_weekend']):>8}")

# Isolate genuine gaps (not holidays or weekends)
genuine_gaps = gaps[~gaps['is_holiday'] & ~gaps['is_weekend']]
print(f"\nGenuine gaps (not holidays/weekends): {len(genuine_gaps)} dates")

if genuine_gaps.empty:
    print("\nAll partial dates are accounted for by NSE holidays or weekends.")
    print("No fix needed!")
    sys.exit(0)

print("\nDates to fix:", genuine_gaps['date_str'].tolist())

# Find the full date range to re-fetch
fix_start = genuine_gaps['Date'].min() - pd.Timedelta(days=5)  # go back a few days for metric context
fix_end = genuine_gaps['Date'].max()

print(f"\nStep 2: Re-fetching data from yfinance ({fix_start.date()} to {fix_end.date()})...")
RAW_FILE = r'c:\userdata\repo\NSEStockDataAnalysis\nse_stock_data_fix.csv'
fetch_data.fetch_stock_data(RAW_FILE, start_date=fix_start)

if not os.path.exists(RAW_FILE):
    print("No data returned from yfinance.")
    sys.exit(1)

new_df = pd.read_csv(RAW_FILE)
new_df['Date'] = pd.to_datetime(new_df['Date'])
# Only keep the gap dates
new_df = new_df[new_df['Date'].isin(genuine_gaps['Date'])]
print(f"Got {len(new_df)} rows for the gap dates from yfinance.")

if new_df.empty:
    print("No new data for gap dates. They may be genuine market holidays not in our list.")
    sys.exit(0)

# Step 3: Load base CSV, merge, recompute metrics
print("\nStep 3: Loading base CSV and computing metrics...")
base_df = pd.read_csv(DATA_FILE)
base_df['Date'] = pd.to_datetime(base_df['Date'])

# Remove gap dates from base and replace with fresh data
base_df = base_df[~base_df['Date'].isin(new_df['Date'].unique())]
raw_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
merged_df = pd.concat([
    base_df[[c for c in raw_cols if c in base_df.columns]],
    new_df[[c for c in raw_cols if c in new_df.columns]]
], ignore_index=True)
merged_df.sort_values(by=['Ticker', 'Date'], inplace=True)

print("Computing metrics (this takes a minute)...")
merged_df['RSI_14'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.rsi(x, length=14))
for w in [3, 20, 100, 200]:
    merged_df[f'{w}DMA'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=w))
for w in [3, 20, 100, 200]:
    def vwma(g, ww=w):
        r = ta.vwma(g['Close'], g['Volume'], length=ww)
        if r is not None: return r
        return (g['Close'] * g['Volume']).rolling(ww, min_periods=1).sum() / g['Volume'].rolling(ww, min_periods=1).sum()
    merged_df[f'{w}VWMA'] = merged_df.groupby('Ticker', group_keys=False).apply(vwma)

merged_df['HA_Close'] = (merged_df['Open'] + merged_df['High'] + merged_df['Low'] + merged_df['Close']) / 4
init_ha = (merged_df['Open'] + merged_df['Close']) / 2
x = merged_df.groupby('Ticker')['HA_Close'].shift(1).fillna(init_ha)
merged_df['HA_Open'] = x.groupby(merged_df['Ticker']).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
for col in [f'{w}DMA' for w in [3, 20, 100, 200]] + ['RSI_14']:
    merged_df[f'{col}_SLOPE'] = merged_df.groupby('Ticker')[col].transform(lambda x: calculate_angle(x, 3))

# Step 4: Extract only the gap rows and upsert to DB
gap_rows = merged_df[merged_df['Date'].isin(genuine_gaps['Date'])]
print(f"\nStep 4: Upserting {len(gap_rows)} fixed rows to CockroachDB...")
count = db_utils.upsert_metrics(gap_rows)
print(f"Upserted {count} rows.")

# Step 5: Save updated CSV
print("\nStep 5: Saving updated CSV...")
merged_df.sort_values(by=['Ticker', 'Date'], ascending=[True, False], inplace=True)
merged_df.to_csv(DATA_FILE, index=False)
print("Done! CSV saved.")

# Verify
print("\nVerification — ticker counts for fixed dates:")
verify_df = pd.read_sql(
    f'SELECT "Date", COUNT(DISTINCT "Ticker") as ticker_count FROM stock_metrics '
    f'WHERE "Date" = ANY(ARRAY[{",".join([chr(39)+d+chr(39) for d in genuine_gaps["date_str"]])}]::DATE[]) '
    f'GROUP BY "Date" ORDER BY "Date"',
    engine
)
print(verify_df.to_string(index=False))
