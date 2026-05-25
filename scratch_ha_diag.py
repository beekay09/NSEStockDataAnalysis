import pandas as pd

df = pd.read_csv('nse_stock_data_with_metrics_v2.csv')
print('HA columns:', [c for c in df.columns if 'HA' in c or 'ha' in c.lower()])
print('Total rows:', len(df))
print('Unique tickers:', df['Ticker'].nunique())

if 'HA_Open' not in df.columns or 'HA_Close' not in df.columns:
    print('\n*** HA_Open / HA_Close columns are MISSING! ***')
    print('All columns:', list(df.columns))
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

# Check last 3 rows for a few tickers
print('\n--- Sample HA data (5 tickers, last 3 rows) ---')
for t in df['Ticker'].unique()[:5]:
    sub = df[df['Ticker'] == t].tail(3)
    print(f'\n  {t}:')
    for _, r in sub.iterrows():
        ha_o = r['HA_Open']
        ha_c = r['HA_Close']
        color = 'RED' if ha_c < ha_o else 'GREEN'
        body = abs(ha_c - ha_o)
        print(f'    {r["Date"].date()}  O={r["Open"]:.2f} C={r["Close"]:.2f}  HA_O={ha_o:.2f} HA_C={ha_c:.2f}  [{color}] body={body:.2f}')

# Count how many tickers have a red->green turn on latest day
turn_count = 0
potential_count = 0
all_red_count = 0
all_green_count = 0

for t in df['Ticker'].unique():
    sub = df[df['Ticker'] == t].tail(3).reset_index(drop=True)
    if len(sub) < 3:
        continue
    
    prev2_red = sub.loc[0, 'HA_Close'] < sub.loc[0, 'HA_Open']
    prev_red = sub.loc[1, 'HA_Close'] < sub.loc[1, 'HA_Open']
    curr_red = sub.loc[2, 'HA_Close'] < sub.loc[2, 'HA_Open']
    curr_green = sub.loc[2, 'HA_Close'] > sub.loc[2, 'HA_Open']
    
    if curr_green:
        all_green_count += 1
    if curr_red:
        all_red_count += 1
    
    # Actual turn: prev red, curr green
    if prev_red and curr_green:
        turn_count += 1
    
    # Potential: last 2 red, body shrinking
    if prev_red and curr_red:
        prev_body = sub.loc[1, 'HA_Open'] - sub.loc[1, 'HA_Close']
        curr_body = sub.loc[2, 'HA_Open'] - sub.loc[2, 'HA_Close']
        if curr_body < prev_body:
            potential_count += 1

print(f'\n--- Summary ---')
print(f'Current GREEN candles: {all_green_count}')
print(f'Current RED candles:   {all_red_count}')
print(f'Red->Green turns:     {turn_count}')
print(f'Potential turns (shrinking red): {potential_count}')
