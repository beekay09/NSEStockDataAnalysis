import db_utils
import pandas as pd

engine = db_utils.get_engine()
df = pd.read_sql('SELECT "Ticker", COUNT(*) as c FROM stock_metrics GROUP BY "Ticker" ORDER BY "Ticker"', engine)
print(df.tail(10))
print('Total:', df['c'].sum())
