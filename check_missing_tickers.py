"""Find which tickers are missing on specific problem dates"""
import sys
import pandas as pd
sys.path.insert(0, r'c:\userdata\repo\NSEStockDataAnalysis')
import db_utils

engine = db_utils.get_engine()

# Get full ticker list from DB
all_tickers = pd.read_sql('SELECT DISTINCT "Ticker" FROM stock_metrics ORDER BY "Ticker"', engine)['Ticker'].tolist()
print(f"Total unique tickers in DB: {len(all_tickers)}")

# Problem dates to investigate
for date_str in ['2026-01-15', '2026-05-01']:
    tickers_on_date = pd.read_sql(
        f'SELECT DISTINCT "Ticker" FROM stock_metrics WHERE "Date" = \'{date_str}\'',
        engine
    )['Ticker'].tolist()
    missing = sorted(set(all_tickers) - set(tickers_on_date))
    print(f"\n{date_str}: {len(tickers_on_date)} tickers. Missing {len(missing)}:")
    print(", ".join(missing))
