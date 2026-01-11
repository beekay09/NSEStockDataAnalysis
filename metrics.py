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

def calculate_relative_strength(stock_df, benchmark_df, window=50):
    """
    Calculate Relative Strength vs Benchmark.
    RS = (Stock / Benchmark) * 100
    Also calculates an 'RS Rating' proxy using the slope of the RS line.
    """
    # Align data by Date
    stock_df = stock_df.set_index('Date')
    benchmark_df = benchmark_df.set_index('Date')
    
    # Inner join to ensure same dates
    combined = stock_df[['Close']].join(benchmark_df[['Close']], lsuffix='_Stock', rsuffix='_Bench').dropna()
    
    # Calculate Ratio
    combined['RS_Ratio'] = (combined['Close_Stock'] / combined['Close_Bench']) * 100
    
    # Calculate RS Slope (Momentum of RS)
    # We use the same slope logic: 100-day rolling slope of the RS Ratio?
    # Or just current slope of line. Let's stick to our project's standard 200DMA/20DMA Slope logic.
    # For RS, a simplified 20-day slope of the Ratio is a good "Strength" indicator.
    
    # Calculate RS_SMA (e.g. 50 period)
    combined['RS_MA'] = combined['RS_Ratio'].rolling(window=window).mean()
    
    # Calculate Slope
    combined['RS_Slope'] = np.degrees(np.arctan(combined['RS_Ratio'].pct_change(20).fillna(0) * 100))
    
    # Reset index to return
    combined = combined.reset_index()
    return combined

def calculate_supertrend(df, period=10, multiplier=3):
    """
    Calculate Supertrend Indicator.
    Returns DataFrame with 'Supertrend', 'Direction' (1=Up, -1=Down).
    """
    st_df = df.copy()
    
    # Calculate TR (True Range)
    st_df['H-L'] = st_df['High'] - st_df['Low']
    st_df['H-PC'] = abs(st_df['High'] - st_df['Close'].shift(1))
    st_df['L-PC'] = abs(st_df['Low'] - st_df['Close'].shift(1))
    st_df['TR'] = st_df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate ATR
    st_df['ATR'] = st_df['TR'].ewm(alpha=1/period, min_periods=period).mean()
    
    # Basic Bands
    st_df['Basic_Upper'] = (st_df['High'] + st_df['Low']) / 2 + (multiplier * st_df['ATR'])
    st_df['Basic_Lower'] = (st_df['High'] + st_df['Low']) / 2 - (multiplier * st_df['ATR'])
    
    # Final Bands
    st_df['Final_Upper'] = st_df['Basic_Upper']
    st_df['Final_Lower'] = st_df['Basic_Lower']
    st_df['Supertrend'] = np.nan
    st_df['Direction'] = 1 # 1: Uptrend, -1: Downtrend
    
    # Iterative calculation for Supertrend logic
    # We need to iterate because current value depends on previous trend
    
    # Convert to numpy arrays for speed
    close = st_df['Close'].values
    basic_upper = st_df['Basic_Upper'].values
    basic_lower = st_df['Basic_Lower'].values
    
    final_upper = np.zeros(len(st_df))
    final_lower = np.zeros(len(st_df))
    supertrend = np.full(len(st_df), np.nan)
    direction = np.zeros(len(st_df))
    
    # Initialize first valid index
    # (Assuming first few are NaN due to ATR)
    
    for i in range(period, len(st_df)):
        # Final Upper
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Final Lower
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Trend Direction
        # Assuming initial direction based on first calc
        if i == period:
            direction[i] = 1 if close[i] > final_upper[i] else -1
            supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
        else:
            prev_dir = direction[i-1]
            if prev_dir == 1:
                if close[i] < final_lower[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
            else: # prev_dir == -1
                if close[i] > final_upper[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_upper[i]
                    
    st_df['Supertrend'] = supertrend
    st_df['Direction'] = direction
    
    return st_df
