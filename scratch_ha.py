import pandas as pd

data = {
    'Ticker': ['A', 'A', 'A', 'B', 'B'],
    'Open': [10, 11, 12, 20, 22],
    'High': [15, 16, 17, 25, 27],
    'Low': [9, 10, 11, 19, 21],
    'Close': [14, 15, 16, 24, 26]
}
df = pd.DataFrame(data)

df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
initial_ha_open = (df['Open'] + df['Close']) / 2
x = df.groupby('Ticker')['HA_Close'].shift(1).fillna(initial_ha_open)
df['HA_Open'] = df.groupby('Ticker', group_keys=False).apply(lambda g: x.loc[g.index].ewm(alpha=0.5, adjust=False).mean())

print(df[['Ticker', 'Open', 'Close', 'HA_Open', 'HA_Close']])
