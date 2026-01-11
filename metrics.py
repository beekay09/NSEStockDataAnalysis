import pandas as pd
import numpy as np

def calculate_beta_metrics(df):
    """Calculate Beta for each stock relative to the equal-weighted market index."""
    # 1. Create a synthetic "Market" index (Equal Weighted)
    market_returns = df.groupby('Date')['Daily_Return'].mean()
    market_variance = market_returns.var()
    
    beta_map = {}
    
    # Calculate Beta for each stock
    # Beta = Cov(Stock, Market) / Var(Market)
    for ticker in df['Ticker'].unique():
        stock_returns = df[df['Ticker'] == ticker].set_index('Date')['Daily_Return']
        
        # Align dates
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        
        if len(aligned_data) > 30: # Minimum data points
            covariance = aligned_data.iloc[:, 0].cov(aligned_data.iloc[:, 1])
            beta = covariance / market_variance
            beta_map[ticker] = beta
        else:
            beta_map[ticker] = np.nan
            
    return beta_map

def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candles from a DataFrame with Open, High, Low, Close."""
    ha_df = df.copy()
    
    # HA Close
    ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # HA Open
    # Initialize with the first row's data
    ha_df['Open'] = 0.0
    # We need to iterate because HA Open depends on previous HA Open
    # Using a list for speed
    open_prices = df['Open'].values
    close_prices = df['Close'].values
    ha_open = [ (open_prices[0] + close_prices[0]) / 2 ]
    
    ha_close_values = ha_df['Close'].values
    
    for i in range(1, len(df)):
        prev_open = ha_open[-1]
        prev_close = ha_close_values[i-1]
        current_open = (prev_open + prev_close) / 2
        ha_open.append(current_open)
        
    ha_df['Open'] = ha_open
    
    # HA High and Low
    ha_df['High'] = ha_df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['Low'] = ha_df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    bb_df = df.copy()
    bb_df['SMA'] = bb_df['Close'].rolling(window=window).mean()
    bb_df['STD'] = bb_df['Close'].rolling(window=window).std()
    bb_df['Upper'] = bb_df['SMA'] + (bb_df['STD'] * num_std)
    bb_df['Lower'] = bb_df['SMA'] - (bb_df['STD'] * num_std)
    return bb_df

def calculate_volume_profile(df, bins=50):
    """Calculate Volume Profile."""
    # Use Close price for binning
    price = df['Close']
    volume = df['Volume']
    
    # Create bins
    hist, bin_edges = np.histogram(price, bins=bins, weights=volume)
    
    # Center of bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    vp_df = pd.DataFrame({'Price': bin_centers, 'Volume': hist})
    return vp_df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal Line, and Histogram."""
    macd_df = df.copy()
    macd_df['EMA_12'] = macd_df['Close'].ewm(span=fast, adjust=False, min_periods=1).mean()
    macd_df['EMA_26'] = macd_df['Close'].ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_df['MACD'] = macd_df['EMA_12'] - macd_df['EMA_26']
    macd_df['Signal'] = macd_df['MACD'].ewm(span=signal, adjust=False, min_periods=1).mean()
    macd_df['Hist'] = macd_df['MACD'] - macd_df['Signal']
    return macd_df
