"""Fix missing tickers on specific trading dates by re-fetching from yfinance"""
import os, sys
import pandas as pd
import pandas_ta as ta
sys.path.insert(0, r'c:\userdata\repo\NSEStockDataAnalysis')
import db_utils
import fetch_data
from add_metrics import calculate_angle

DATA_FILE = r'c:\userdata\repo\NSEStockDataAnalysis\nse_stock_data_with_metrics_v2.csv'

# Only fix genuine trading days with missing tickers
FIX_DATES = ['2026-01-15']
MISSING = {
    '2026-01-15': ['AJANTPHARM.NS','ALKEM.NS','AUROPHARMA.NS','BHARATFORG.NS',
                   'GRANULES.NS','KANSAINER.NS','MRF.NS','PAGEIND.NS',
                   'PHOENIXLTD.NS','POLICYBZR.NS','PREMIER.NS','SIEMENS.NS',
                   'SOLARINDS.NS','TATSILV.NS']
}

engine = db_utils.get_engine()

for date_str in FIX_DATES:
    fix_date = pd.Timestamp(date_str)
    tickers = MISSING[date_str]
    print(f"\n=== Fixing {date_str} — fetching {len(tickers)} tickers ===")

    # Fetch a window around the fix date for metric computation context
    fetch_start = fix_date - pd.Timedelta(days=10)
    fetch_end   = fix_date + pd.Timedelta(days=1)

    import yfinance as yf
    raw = yf.download(
        tickers, start=fetch_start, end=fetch_end,
        group_by='ticker', auto_adjust=True, threads=True
    )

    if raw.empty:
        print(f"No data returned for {date_str}")
        continue

    # Flatten multi-index
    rows = []
    for ticker in tickers:
        symbol = ticker.replace('.NS', '')
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                t_df = raw[ticker].dropna(how='all').copy()
            else:
                t_df = raw.dropna(how='all').copy()
            t_df = t_df.reset_index()
            t_df.columns = [c[0] if isinstance(c, tuple) else c for c in t_df.columns]
            t_df['Ticker'] = symbol
            t_df.rename(columns={'index': 'Date', 'Datetime': 'Date'}, inplace=True, errors='ignore')
            # Only keep the fix date
            t_df['Date'] = pd.to_datetime(t_df['Date'])
            t_df = t_df[t_df['Date'].dt.date == fix_date.date()]
            rows.append(t_df)
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    if not rows:
        print("No rows to fix.")
        continue

    new_df = pd.concat(rows, ignore_index=True)
    new_df = new_df[['Date','Ticker','Open','High','Low','Close','Volume']]
    print(f"Got {len(new_df)} new rows from yfinance for {date_str}")

    # Merge with CSV base data
    base_df = pd.read_csv(DATA_FILE)
    base_df['Date'] = pd.to_datetime(base_df['Date'])

    # Remove existing rows for these tickers on this date (to avoid duplicates)
    mask = (base_df['Date'].dt.date == fix_date.date()) & (base_df['Ticker'].isin(new_df['Ticker']))
    base_df = base_df[~mask]

    raw_cols = ['Date','Ticker','Open','High','Low','Close','Volume']
    merged_df = pd.concat([
        base_df[[c for c in raw_cols if c in base_df.columns]],
        new_df[[c for c in raw_cols if c in new_df.columns]]
    ], ignore_index=True).sort_values(['Ticker','Date'])

    # Recompute metrics
    print("Recomputing metrics...")
    merged_df['RSI_14'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.rsi(x, length=14))
    for w in [3,20,100,200]:
        merged_df[f'{w}DMA'] = merged_df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=w))
    for w in [3,20,100,200]:
        def vwma(g, ww=w):
            r = ta.vwma(g['Close'], g['Volume'], length=ww)
            return r if r is not None else (g['Close']*g['Volume']).rolling(ww,min_periods=1).sum()/g['Volume'].rolling(ww,min_periods=1).sum()
        merged_df[f'{w}VWMA'] = merged_df.groupby('Ticker', group_keys=False).apply(vwma)
    merged_df['HA_Close'] = (merged_df['Open']+merged_df['High']+merged_df['Low']+merged_df['Close'])/4
    init_ha = (merged_df['Open']+merged_df['Close'])/2
    x = merged_df.groupby('Ticker')['HA_Close'].shift(1).fillna(init_ha)
    merged_df['HA_Open'] = x.groupby(merged_df['Ticker']).transform(lambda s: s.ewm(alpha=0.5,adjust=False).mean())
    for col in [f'{w}DMA' for w in [3,20,100,200]]+['RSI_14']:
        merged_df[f'{col}_SLOPE'] = merged_df.groupby('Ticker')[col].transform(lambda x: calculate_angle(x,3))

    # Extract only the fix-date rows for upserting
    fix_rows = merged_df[merged_df['Date'].dt.date == fix_date.date()]
    print(f"Upserting {len(fix_rows)} rows to CockroachDB...")
    count = db_utils.upsert_metrics(fix_rows)
    print(f"Done. Upserted {count} rows.")

    # Save updated CSV
    merged_df.sort_values(['Ticker','Date'], ascending=[True,False], inplace=True)
    merged_df.to_csv(DATA_FILE, index=False)
    print("CSV saved.")

    # Verify
    verify = pd.read_sql(
        f'SELECT COUNT(DISTINCT "Ticker") as cnt FROM stock_metrics WHERE "Date" = \'{date_str}\'',
        engine
    )
    print(f"Verification: {date_str} now has {verify['cnt'].iloc[0]} tickers in DB")
