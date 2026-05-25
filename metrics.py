import pandas as pd
import numpy as np
import pandas_ta as ta

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
    """Calculate Heikin Ashi candles using pandas_ta."""
    # pandas_ta has a ha() method, but it requires the DataFrame to have
    # open/high/low/close columns. We'll use it via the DataFrame accessor.
    ha_df = df.copy()
    
    # pandas_ta expects lowercase column names for its accessor
    temp_df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    ha_result = temp_df.ta.ha()
    
    if ha_result is not None and not ha_result.empty:
        # Column names vary by pandas_ta version, so find them dynamically
        ha_cols = ha_result.columns.tolist()
        open_col = [c for c in ha_cols if 'open' in c.lower()][0]
        high_col = [c for c in ha_cols if 'high' in c.lower()][0]
        low_col = [c for c in ha_cols if 'low' in c.lower()][0]
        close_col = [c for c in ha_cols if 'close' in c.lower()][0]
        ha_df['Open'] = ha_result[open_col].values
        ha_df['High'] = ha_result[high_col].values
        ha_df['Low'] = ha_result[low_col].values
        ha_df['Close'] = ha_result[close_col].values
    else:
        # Fallback to manual calculation if pandas_ta ha() fails
        ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_df['Open'] = 0.0
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
        ha_df['High'] = ha_df[['High', 'Open', 'Close']].max(axis=1)
        ha_df['Low'] = ha_df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands using pandas_ta."""
    bb_df = df.copy()
    
    # pandas_ta bbands returns: BBL, BBM, BBU, BBB, BBP
    bb_result = ta.bbands(bb_df['Close'], length=window, std=num_std)
    
    if bb_result is not None and not bb_result.empty:
        # Column names vary by pandas_ta version, so match by prefix
        bbm_col = [c for c in bb_result.columns if c.startswith('BBM_')][0]
        bbu_col = [c for c in bb_result.columns if c.startswith('BBU_')][0]
        bbl_col = [c for c in bb_result.columns if c.startswith('BBL_')][0]
        bb_df['SMA'] = bb_result[bbm_col].values
        bb_df['Upper'] = bb_result[bbu_col].values
        bb_df['Lower'] = bb_result[bbl_col].values
    else:
        # Fallback
        bb_df['SMA'] = bb_df['Close'].rolling(window=window).mean()
        bb_df['STD'] = bb_df['Close'].rolling(window=window).std()
        bb_df['Upper'] = bb_df['SMA'] + (bb_df['STD'] * num_std)
        bb_df['Lower'] = bb_df['SMA'] - (bb_df['STD'] * num_std)
    
    return bb_df

def calculate_volume_profile(df, bins=50):
    """Calculate Volume Profile. (No pandas_ta equivalent — custom logic retained.)"""
    price = df['Close']
    volume = df['Volume']
    
    hist, bin_edges = np.histogram(price, bins=bins, weights=volume)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    vp_df = pd.DataFrame({'Price': bin_centers, 'Volume': hist})
    return vp_df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal Line, and Histogram using pandas_ta."""
    macd_df = df.copy()
    
    macd_result = ta.macd(macd_df['Close'], fast=fast, slow=slow, signal=signal)
    
    if macd_result is not None and not macd_result.empty:
        # Column names vary by pandas_ta version, so match by prefix
        macd_col = [c for c in macd_result.columns if c.startswith('MACD_')][0]
        signal_col = [c for c in macd_result.columns if c.startswith('MACDs_')][0]
        hist_col = [c for c in macd_result.columns if c.startswith('MACDh_')][0]
        macd_df['MACD'] = macd_result[macd_col].values
        macd_df['Signal'] = macd_result[signal_col].values
        macd_df['Hist'] = macd_result[hist_col].values
    else:
        # Fallback
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
    
    # Calculate RS_SMA using pandas_ta
    rs_ma = ta.sma(combined['RS_Ratio'], length=window)
    combined['RS_MA'] = rs_ma.values if rs_ma is not None else combined['RS_Ratio'].rolling(window=window).mean()
    
    # Calculate Slope
    combined['RS_Slope'] = np.degrees(np.arctan(combined['RS_Ratio'].pct_change(20).fillna(0) * 100))
    
    # Reset index to return
    combined = combined.reset_index()
    return combined

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX) and DI+/DI- using pandas_ta."""
    adx_df = df.copy()
    
    adx_result = ta.adx(adx_df['High'], adx_df['Low'], adx_df['Close'], length=period)
    
    if adx_result is not None and not adx_result.empty:
        # Column names vary by pandas_ta version, so match by prefix
        adx_col = [c for c in adx_result.columns if c.startswith('ADX_')][0]
        dmp_col = [c for c in adx_result.columns if c.startswith('DMP_')][0]
        dmn_col = [c for c in adx_result.columns if c.startswith('DMN_')][0]
        adx_df['ADX'] = adx_result[adx_col].values
        adx_df['+DI'] = adx_result[dmp_col].values
        adx_df['-DI'] = adx_result[dmn_col].values
    else:
        # Fallback to manual calculation
        adx_df['H-L'] = adx_df['High'] - adx_df['Low']
        adx_df['H-PC'] = abs(adx_df['High'] - adx_df['Close'].shift(1))
        adx_df['L-PC'] = abs(adx_df['Low'] - adx_df['Close'].shift(1))
        adx_df['TR'] = adx_df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        adx_df['UpMove'] = adx_df['High'] - adx_df['High'].shift(1)
        adx_df['DownMove'] = adx_df['Low'].shift(1) - adx_df['Low']
        
        adx_df['+DM'] = np.where((adx_df['UpMove'] > adx_df['DownMove']) & (adx_df['UpMove'] > 0), adx_df['UpMove'], 0)
        adx_df['-DM'] = np.where((adx_df['DownMove'] > adx_df['UpMove']) & (adx_df['DownMove'] > 0), adx_df['DownMove'], 0)
        
        alpha = 1 / period
        adx_df['TR_Smooth'] = adx_df['TR'].ewm(alpha=alpha, adjust=False).mean()
        adx_df['+DM_Smooth'] = adx_df['+DM'].ewm(alpha=alpha, adjust=False).mean()
        adx_df['-DM_Smooth'] = adx_df['-DM'].ewm(alpha=alpha, adjust=False).mean()
        
        adx_df['+DI'] = 100 * (adx_df['+DM_Smooth'] / adx_df['TR_Smooth'])
        adx_df['-DI'] = 100 * (adx_df['-DM_Smooth'] / adx_df['TR_Smooth'])
        
        adx_df['DX'] = 100 * abs(adx_df['+DI'] - adx_df['-DI']) / (adx_df['+DI'] + adx_df['-DI'])
        adx_df['ADX'] = adx_df['DX'].ewm(alpha=alpha, adjust=False).mean()
    
    return adx_df[['Date', 'ADX', '+DI', '-DI']]
