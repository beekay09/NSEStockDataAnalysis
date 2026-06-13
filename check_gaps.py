"""Check all dates that don't have exactly 222 tickers"""
import sys
import pandas as pd
sys.path.insert(0, r'c:\userdata\repo\NSEStockDataAnalysis')
import db_utils

# NSE holidays
NSE_HOLIDAYS = {
    "2024-10-02","2024-11-01","2024-11-15","2024-12-25",
    "2025-01-26","2025-02-26","2025-03-14","2025-03-31",
    "2025-04-10","2025-04-14","2025-04-18","2025-05-01",
    "2025-06-07","2025-08-15","2025-08-27","2025-10-02",
    "2025-10-20","2025-10-21","2025-10-24","2025-11-05",
    "2025-12-25",
    "2026-01-26","2026-02-26","2026-03-20","2026-03-30",
    "2026-04-02","2026-04-06","2026-04-10","2026-04-14",
    "2026-05-01","2026-05-25",
}

engine = db_utils.get_engine()
df = pd.read_sql(
    'SELECT "Date", COUNT(DISTINCT "Ticker") as ticker_count '
    'FROM stock_metrics GROUP BY "Date" ORDER BY "Date"',
    engine
)
df['Date'] = pd.to_datetime(df['Date'])
df['date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
df['is_holiday'] = df['date_str'].isin(NSE_HOLIDAYS)
df['is_weekend'] = df['Date'].dt.dayofweek >= 5

# All dates not at 222
gaps = df[df['ticker_count'] < 222].copy()
print(f"{'Date':<15} {'Count':>6} {'Holiday':>8} {'Weekend':>8}")
print("-" * 45)
for _, row in gaps.iterrows():
    print(f"{row['date_str']:<15} {row['ticker_count']:>6} {str(row['is_holiday']):>8} {str(row['is_weekend']):>8}")

print(f"\nTotal: {len(gaps)} dates with fewer than 222 stocks")
