"""
Backfill historical data for tickers that were added to stocks.txt later
than others, leaving gaps in older dates.

Logic:
1. Find the global earliest date in the DB (what all stocks should go back to)
2. Find tickers whose earliest date in the DB is AFTER the global min date
3. For each such ticker, fetch the missing history from yfinance
4. Recompute metrics and upsert to DB + save updated CSV
"""
import os, sys
import pandas as pd
import pandas_ta as ta
import yfinance as yf

sys.path.insert(0, r'c:\userdata\repo\NSEStockDataAnalysis')
import db_utils
from add_metrics import calculate_angle

DATA_FILE = r'c:\userdata\repo\NSEStockDataAnalysis\nse_stock_data_with_metrics_v2.csv'

engine = db_utils.get_engine()

# Step 1: Find global min date and per-ticker min date
print("Querying DB for per-ticker earliest dates...")
ticker_min_df = pd.read_sql(
    'SELECT "Ticker", MIN("Date") as first_date FROM stock_metrics GROUP BY "Ticker" ORDER BY first_date',
    engine
)
ticker_min_df['first_date'] = pd.to_datetime(ticker_min_df['first_date'])

global_min_date = ticker_min_df['first_date'].min()
print(f"Global earliest date in DB: {global_min_date.date()}")
print(f"Total tickers: {len(ticker_min_df)}")

# Step 2: Find tickers that start AFTER the global min (i.e. missing older history)
# Allow 5 days tolerance (weekends / holidays at the start of the range)
tolerance = pd.Timedelta(days=7)
late_starters = ticker_min_df[ticker_min_df['first_date'] > global_min_date + tolerance].copy()
late_starters = late_starters.sort_values('first_date')

print(f"\nTickers missing older history: {len(late_starters)}")
print(f"\n{'Ticker':<20} {'First Date in DB':<20} {'Missing from':<20}")
print("-" * 60)
for _, row in late_starters.iterrows():
    missing_from = global_min_date.date()
    print(f"{row['Ticker']:<20} {str(row['first_date'].date()):<20} {str(missing_from):<20}")

if late_starters.empty:
    print("\nAll tickers have complete history. Nothing to backfill!")
    sys.exit(0)

# Step 3: Fetch missing history for each late-starting ticker from yfinance
yf_symbols = [t + '.NS' for t in late_starters['Ticker'].tolist()]
earliest_needed = global_min_date
# Fetch up to the latest first-date to ensure we cover all gaps
latest_first_date = late_starters['first_date'].max()

print(f"\nFetching data from {earliest_needed.date()} to {latest_first_date.date()} for {len(yf_symbols)} tickers...")
raw = yf.download(
    yf_symbols,
    start=earliest_needed,
    end=latest_first_date + pd.Timedelta(days=1),
    group_by='ticker',
    auto_adjust=True,
    threads=True
)

if raw.empty:
    print("No data returned from yfinance.")
    sys.exit(1)

# Step 4: Flatten multi-index into a clean DataFrame
rows = []
for sym in yf_symbols:
    ticker = sym.replace('.NS', '')
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            t_df = raw[sym].dropna(how='all').reset_index()
        else:
            t_df = raw.dropna(how='all').reset_index()
        t_df.columns = [c[0] if isinstance(c, tuple) else c for c in t_df.columns]
        t_df.rename(columns={'index': 'Date', 'Datetime': 'Date', 'Price': 'Date'}, inplace=True, errors='ignore')
        t_df['Date'] = pd.to_datetime(t_df['Date'])
        # Only keep dates BEFORE the ticker's first date in DB (the truly missing part)
        ticker_first = late_starters[late_starters['Ticker'] == ticker]['first_date'].values[0]
        t_df = t_df[t_df['Date'] < pd.Timestamp(ticker_first)]
        t_df['Ticker'] = ticker
        rows.append(t_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])
        print(f"  {ticker}: fetched {len(t_df)} new rows (up to {pd.Timestamp(ticker_first).date()})")
    except Exception as e:
        print(f"  Skipping {sym}: {e}")

if not rows:
    print("No new rows fetched.")
    sys.exit(0)

new_df = pd.concat(rows, ignore_index=True)
new_df = new_df.dropna(subset=['Open', 'Close'])
print(f"\nTotal new rows fetched: {len(new_df)}")
print(f"Date range: {new_df['Date'].min().date()} to {new_df['Date'].max().date()}")

# Step 5: Load CSV, merge new rows, recompute metrics
print("\nLoading base data from CSV...")
base_df = pd.read_csv(DATA_FILE)
base_df['Date'] = pd.to_datetime(base_df['Date'])

# Remove any overlap (shouldn't be any, but safety first)
overlap_mask = (
    base_df['Ticker'].isin(new_df['Ticker']) &
    base_df['Date'].isin(new_df['Date'])
)
base_df = base_df[~overlap_mask]

raw_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
merged_df = pd.concat([
    base_df[[c for c in raw_cols if c in base_df.columns]],
    new_df[raw_cols]
], ignore_index=True).sort_values(['Ticker', 'Date'])

# Step 6: Recompute all metrics
print("Recomputing metrics (this takes ~1 minute)...")
merged_df['RSI_14'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.rsi(x, length=14))
for w in [3, 20, 100, 200]:
    merged_df[f'{w}DMA'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=w))
for w in [3, 20, 100, 200]:
    def vwma(g, ww=w):
        r = ta.vwma(g['Close'], g['Volume'], length=ww)
        return r if r is not None else (g['Close']*g['Volume']).rolling(ww, min_periods=1).sum() / g['Volume'].rolling(ww, min_periods=1).sum()
    merged_df[f'{w}VWMA'] = merged_df.groupby('Ticker', group_keys=False).apply(vwma)

merged_df['HA_Close'] = (merged_df['Open'] + merged_df['High'] + merged_df['Low'] + merged_df['Close']) / 4
init_ha = (merged_df['Open'] + merged_df['Close']) / 2
x = merged_df.groupby('Ticker')['HA_Close'].shift(1).fillna(init_ha)
merged_df['HA_Open'] = x.groupby(merged_df['Ticker']).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
for col in [f'{w}DMA' for w in [3, 20, 100, 200]] + ['RSI_14']:
    merged_df[f'{col}_SLOPE'] = merged_df.groupby('Ticker')[col].transform(lambda x: calculate_angle(x, 3))

# Step 7: Upsert only the newly fetched rows to DB
upsert_df = merged_df[
    merged_df['Ticker'].isin(late_starters['Ticker']) &
    merged_df['Date'].isin(new_df['Date'])
]
print(f"\nUpserting {len(upsert_df)} rows to CockroachDB...")
count = db_utils.upsert_metrics(upsert_df)
print(f"Done. Upserted {count} rows.")

# Step 8: Save updated CSV
print("Saving updated CSV...")
merged_df.sort_values(['Ticker', 'Date'], ascending=[True, False], inplace=True)
merged_df.to_csv(DATA_FILE, index=False)
print("CSV saved.")

# Step 9: Verify
print("\nVerification — new earliest dates per ticker:")
verify_df = pd.read_sql(
    'SELECT "Ticker", MIN("Date") as first_date, COUNT(*) as total_rows '
    'FROM stock_metrics '
    f'WHERE "Ticker" = ANY(ARRAY[{",".join([chr(39)+t+chr(39) for t in late_starters["Ticker"].tolist()])}]) '
    'GROUP BY "Ticker" ORDER BY "Ticker"',
    engine
)
print(verify_df.to_string(index=False))
