import pandas as pd
import numpy as np
import metrics as mt

def get_low_rsi_tickers(latest_df, rsi_threshold, min_angle):
    return latest_df[
        (latest_df['RSI_14'] < rsi_threshold) & 
        (latest_df['200DMA_SLOPE'] > min_angle)
    ]['Ticker'].tolist(), "Low RSI"

def get_low_beta_tickers(latest_df, beta_percentile, min_angle):
    beta_threshold = latest_df['Beta'].quantile(beta_percentile)
    return latest_df[
        (latest_df['Beta'] <= beta_threshold) & 
        (latest_df['200DMA_SLOPE'] > min_angle)
    ]['Ticker'].tolist(), "Low Beta"

def get_narrow_band_tickers(latest_df, df, band_pct, lookback_days, min_angle):
    narrow_band_tickers = []
    candidates = latest_df[latest_df['200DMA_SLOPE'] > min_angle]['Ticker'].tolist()
    for ticker in candidates:
        stock_data = df[df['Ticker'] == ticker].sort_values('Date').tail(lookback_days)
        if len(stock_data) < lookback_days:
            continue
        min_price = stock_data['Close'].min()
        max_price = stock_data['Close'].max()
        if (max_price - min_price) / min_price < band_pct:
            narrow_band_tickers.append(ticker)
    return narrow_band_tickers, "Narrow Band"

def get_divergence_slope_tickers(latest_df, min_angle):
    slope_filtered = latest_df[latest_df['200DMA_SLOPE'] > min_angle]
    return slope_filtered.sort_values(by='3DMA_SLOPE', ascending=False).head(10)['Ticker'].tolist(), "Divergence Slope"

def get_high_dma_angle_tickers(latest_df, dma, top_n, min_angle):
    target_col = f"{dma}DMA_SLOPE"
    return latest_df.sort_values(by=target_col, ascending=False).head(top_n)['Ticker'].tolist(), "High DMA Angle"

def get_slope_difference_tickers(latest_df, min_diff, min_200_slope, top_n):
    # Ensure SlopeDiff is calculated
    if 'SlopeDiff' not in latest_df.columns:
        latest_df['SlopeDiff'] = latest_df['20DMA_SLOPE'] - latest_df['200DMA_SLOPE']
        
    mask = (latest_df['SlopeDiff'] >= min_diff) & (latest_df['200DMA_SLOPE'] > min_200_slope)
    return latest_df[mask].sort_values(by='SlopeDiff', ascending=False).head(top_n)['Ticker'].tolist(), "Slope Difference"

def get_actual_crossover_tickers(latest_df, min_angle):
    return latest_df[
        (latest_df['20DMA'] > latest_df['200DMA']) &
        (latest_df['Prev_20DMA'] <= latest_df['Prev_200DMA']) &
        (latest_df['200DMA_SLOPE'] > min_angle)
    ]['Ticker'].tolist(), "Actual Crossover"

def get_potential_crossover_tickers(latest_df, proximity_pct, min_angle):
    mask_potential = (
        (latest_df['20DMA'] < latest_df['200DMA']) &
        (latest_df['200DMA'] != 0) &
        ((abs(latest_df['20DMA'] - latest_df['200DMA']) / latest_df['200DMA']) * 100 < proximity_pct) &
        (latest_df['20DMA'] > latest_df['Prev_20DMA']) &
        (latest_df['200DMA_SLOPE'] > min_angle)
    )
    return latest_df[mask_potential]['Ticker'].tolist(), "Potential Crossover"

def get_volume_shockers_tickers(latest_df, vol_ratio_threshold, min_avg_vol, min_angle):
    shockers_mask = (
        (latest_df['VolumeRatio'] > vol_ratio_threshold) &
        (latest_df['20DayAvgVolume'] > min_avg_vol) &
        (latest_df['200DMA_SLOPE'] > min_angle)
    )
    return latest_df[shockers_mask]['Ticker'].tolist(), "Volume Shockers"

def get_dma_bottoming_tickers(latest_df, dma, min_angle, max_angle, prev_max_angle):
    cur_col = f"{dma}DMA_SLOPE"
    prev_col = f"Prev_{dma}DMA_SLOPE"
    
    bottoming_mask = (
        (latest_df[prev_col] <= prev_max_angle) &
        (latest_df[cur_col] > latest_df[prev_col]) &
        (latest_df[cur_col] >= min_angle) &
        (latest_df[cur_col] <= max_angle)
    )
    return latest_df[bottoming_mask]['Ticker'].tolist(), "DMA Bottoming"


def get_macd_crossover_tickers(latest_df, df, min_angle, lookback_days=3):
    # 1. Identify candidates (Positive 200DMA background)
    candidates = latest_df[latest_df['200DMA_SLOPE'] > min_angle]['Ticker'].tolist()
    
    crossover_tickers = []
    
    # 2. Iterate and check MACD condition
    for ticker in candidates:
        # Get sufficient history for MACD (100 days should be enough)
        stock_data = df[df['Ticker'] == ticker].sort_values('Date').tail(100)
        
        if len(stock_data) < 30:
            continue
            
        macd_data = mt.calculate_macd(stock_data)
        
        # Check Crossover: 
        # Today: MACD > Signal
        # Yesterday: MACD <= Signal
        
        if len(macd_data) < (lookback_days + 2):
            continue
            
        # Check for crossover in the last N days
        # We need enough recent data: lookback + 1 (for previous day compare)
        recent = macd_data.tail(lookback_days + 1)
        
        crossover_found = False
        
        for i in range(1, len(recent)):
            curr = recent.iloc[i]
            prev = recent.iloc[i-1]
            
            if (curr['MACD'] > curr['Signal']) and (prev['MACD'] <= prev['Signal']):
                crossover_found = True
                break
        
        if crossover_found:
            crossover_tickers.append(ticker)
            
    return crossover_tickers, "MACD Crossover"

def get_rs_strong_tickers(latest_df, df, benchmark_df, min_rs_slope=0):
    """
    Identify stocks showing relative strength vs Benchmark.
    Criteria:
    1. RS Line is uptrending (Slope > 0)
    2. RS Ratio > Moving Average (optional, simpler to just check slope)
    """
    strong_rs_tickers = []
    
    # Pre-calculate Benchmark once? No, passed as arg.
    
    for ticker in latest_df['Ticker'].tolist():
        stock_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        # Need sufficient overlap
        if len(stock_data) < 50:
            continue
            
        rs_data = mt.calculate_relative_strength(stock_data, benchmark_df)
        
        # Check if empty (no overlapping dates)
        if rs_data.empty or len(rs_data) < 20:
            continue
            
        # Check current slope
        current_slope = rs_data.iloc[-1]['RS_Slope']
        
        # Check if RS is above its MA (Strength confirmation)
        current_rs = rs_data.iloc[-1]['RS_Ratio']
        current_ma = rs_data.iloc[-1]['RS_MA']
        
        if current_slope > min_rs_slope and current_rs > current_ma:
            strong_rs_tickers.append(ticker)
            
    return strong_rs_tickers, "Relative Strength"

def get_strong_adx_tickers(latest_df, full_df, adx_threshold=25):
    """
    Find tickers with ADX > threshold indicating a strong trend.
    Returns: List of tickers, Documentation Key
    """
    tickers = []
    
    # We need full history to calculate ADX
    # Iterate over all tickers present in latest_df
    candidate_tickers = latest_df['Ticker'].unique()
    
    for ticker in candidate_tickers:
        try:
            stock_df = full_df[full_df['Ticker'] == ticker].sort_values('Date')
            
            # optimization: skip if too short
            if len(stock_df) < 30:
                continue
                
            adx_data = mt.calculate_adx(stock_df)
            
            if adx_data.empty:
                continue
                
            latest_adx = adx_data.iloc[-1]
            
            if latest_adx['ADX'] > adx_threshold:
                tickers.append(ticker)
                
        except Exception as e:
            continue
            
    return tickers, "Strong ADX"
