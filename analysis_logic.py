import pandas as pd
import numpy as np
import metrics as mt

def get_low_rsi_tickers(latest_df, rsi_threshold, min_angle, rsi_turning_up=False, volume_spike=False):
    mask = (
        (latest_df['RSI_14'] < rsi_threshold) & 
        (latest_df['200DMA_SLOPE'] > min_angle)
    )
    
    if rsi_turning_up:
        mask = mask & (latest_df['RSI_14_SLOPE'] > 0)
        
    if volume_spike:
        # Volume > 2x Avg AND Green Candle (Close >= Open)
        mask = mask & (latest_df['VolumeRatio'] > 2.0) & (latest_df['Close'] >= latest_df['Open'])

    return latest_df[mask]['Ticker'].tolist(), "Low RSI"

def get_low_beta_tickers(latest_df, beta_percentile, min_angle, near_200dma=False, rsi_turning_up=False):
    beta_threshold = latest_df['Beta'].quantile(beta_percentile)
    mask = (
        (latest_df['Beta'] <= beta_threshold) & 
        (latest_df['200DMA_SLOPE'] > min_angle)
    )
    
    if near_200dma:
        # Close within 5% of 200DMA
        # abs(Close - 200DMA) / 200DMA <= 0.05
        mask = mask & (abs(latest_df['Close'] - latest_df['200DMA']) / latest_df['200DMA'] <= 0.05)
        
    if rsi_turning_up:
        mask = mask & (latest_df['RSI_14_SLOPE'] > 0)

    return latest_df[mask]['Ticker'].tolist(), "Low Beta"

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

def get_divergence_slope_tickers(latest_df, min_angle, rsi_check=False, volume_check=False, max_rsi=70, min_vol_ratio=1.0):
    slope_filtered = latest_df[latest_df['200DMA_SLOPE'] > min_angle]
    
    if rsi_check:
        slope_filtered = slope_filtered[slope_filtered['RSI_14'] < max_rsi]
        
    if volume_check:
        slope_filtered = slope_filtered[slope_filtered['VolumeRatio'] > min_vol_ratio]

    return slope_filtered.sort_values(by='3DMA_SLOPE', ascending=False).head(10)['Ticker'].tolist(), "Divergence Slope"

def get_high_dma_angle_tickers(latest_df, dma, top_n, min_angle):
    target_col = f"{dma}DMA_SLOPE"
    return latest_df.sort_values(by=target_col, ascending=False).head(top_n)['Ticker'].tolist(), "High DMA Angle"

def get_slope_difference_tickers(latest_df, min_diff, min_200_slope, top_n):
    # Ensure SlopeDiff is calculated
    latest_df['SlopeDiff'] = latest_df['20DMA_SLOPE'] - latest_df['200DMA_SLOPE']
        
    mask = (latest_df['SlopeDiff'] >= min_diff) & (latest_df['200DMA_SLOPE'] > min_200_slope)
    return latest_df[mask].sort_values(by='SlopeDiff', ascending=False).head(top_n)['Ticker'].tolist(), "Slope Difference"

def get_actual_crossover_tickers(latest_df, full_df, min_angle, lookback_days=1):
    """
    Find stocks where 20DMA crossed above 200DMA within the last lookback_days.
    Returns: (list of tickers, dict of {ticker: crossover_date}, doc_key)
    """
    if lookback_days <= 1:
        # Original logic: only check latest day
        mask = (
            (latest_df['20DMA'] > latest_df['200DMA']) &
            (latest_df['Prev_20DMA'] <= latest_df['Prev_200DMA']) &
            (latest_df['200DMA_SLOPE'] > min_angle)
        )
        tickers = latest_df[mask]['Ticker'].tolist()
        # For lookback=1, crossover date is the latest date
        crossover_info = {}
        for t in tickers:
            row = latest_df[latest_df['Ticker'] == t]
            if not row.empty and 'Date' in row.columns:
                crossover_info[t] = row.iloc[0]['Date']
        return tickers, crossover_info, "Actual Crossover"
    
    # Lookback > 1: scan historical data
    candidates = latest_df[latest_df['200DMA_SLOPE'] > min_angle]['Ticker'].tolist()
    crossover_tickers = []
    crossover_info = {}
    
    for ticker in candidates:
        stock_data = full_df[full_df['Ticker'] == ticker].sort_values('Date')
        
        if len(stock_data) < lookback_days + 2:
            continue
        
        recent = stock_data.tail(lookback_days + 1).reset_index(drop=True)
        
        for i in range(1, len(recent)):
            curr_20 = recent.loc[i, '20DMA']
            curr_200 = recent.loc[i, '200DMA']
            prev_20 = recent.loc[i-1, '20DMA']
            prev_200 = recent.loc[i-1, '200DMA']
            
            if (curr_20 > curr_200) and (prev_20 <= prev_200):
                crossover_tickers.append(ticker)
                crossover_info[ticker] = recent.loc[i, 'Date']
                break
    
    return crossover_tickers, crossover_info, "Actual Crossover"

def get_price_dma_crossover_tickers(latest_df, full_df, min_angle, lookback_days=1, dma_period=100):
    """
    Find stocks where Price (Close) crossed above the specified DMA within the last lookback_days.
    Returns: (list of tickers, dict of {ticker: crossover_date}, doc_key)
    """
    dma_col = f'{dma_period}DMA'
    slope_col = f'{dma_period}DMA_SLOPE'
    candidates = latest_df[latest_df[slope_col] > min_angle]['Ticker'].tolist()
    crossover_tickers = []
    crossover_info = {}
    
    for ticker in candidates:
        stock_data = full_df[full_df['Ticker'] == ticker].sort_values('Date')
        
        if len(stock_data) < lookback_days + 2:
            continue
        
        recent = stock_data.tail(lookback_days + 1).reset_index(drop=True)
        
        for i in range(1, len(recent)):
            curr_close = recent.loc[i, 'Close']
            curr_dma = recent.loc[i, dma_col]
            prev_close = recent.loc[i-1, 'Close']
            prev_dma = recent.loc[i-1, dma_col]
            
            if (curr_close > curr_dma) and (prev_close <= prev_dma):
                crossover_tickers.append(ticker)
                crossover_info[ticker] = recent.loc[i, 'Date']
                break
    
    return crossover_tickers, crossover_info, "Actual Crossover"

def get_potential_crossover_tickers(latest_df, proximity_pct, min_angle):
    mask_potential = (
        (latest_df['20DMA'] < latest_df['200DMA']) &
        (latest_df['200DMA'] != 0) &
        ((abs(latest_df['20DMA'] - latest_df['200DMA']) / latest_df['200DMA']) * 100 < proximity_pct) &
        (latest_df['20DMA'] > latest_df['Prev_20DMA']) &
        (latest_df['200DMA_SLOPE'] > min_angle)
    )
    return latest_df[mask_potential]['Ticker'].tolist(), "Potential Crossover"

def get_volume_shockers_tickers(latest_df, vol_ratio_threshold, min_avg_vol, min_angle, sentiment="Both"):
    shockers_mask = (
        (latest_df['VolumeRatio'] > vol_ratio_threshold) &
        (latest_df['20DayAvgVolume'] > min_avg_vol) &
        (latest_df['200DMA_SLOPE'] > min_angle)
    )
    filtered_df = latest_df[shockers_mask]
    
    if sentiment == "Bullish":
        filtered_df = filtered_df[filtered_df['Close'] >= filtered_df['Open']]
    elif sentiment == "Bearish":
        filtered_df = filtered_df[filtered_df['Close'] < filtered_df['Open']]
        
    return filtered_df['Ticker'].tolist(), "Volume Shockers"

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

def get_rs_strong_tickers(latest_df, df, benchmark_df, min_rs_slope=0, min_200dma_angle=0):
    """
    Identify stocks showing relative strength vs Benchmark.
    Criteria:
    1. RS Line is uptrending (Slope > 0)
    2. RS Ratio > Moving Average (optional, simpler to just check slope)
    3. 200DMA Angle > min_200dma_angle
    """
    strong_rs_tickers = []
    
    # Filter by 200DMA Angle first
    filtered_df = latest_df[latest_df['200DMA_SLOPE'] > min_200dma_angle]
    
    for ticker in filtered_df['Ticker'].tolist():
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
            
    return list(set(strong_rs_tickers)), "RS > Benchmark"

def get_slope_crossover_tickers(latest_df, df, lookback_days=3, min_20dma_slope=0):
    """
    Identify stocks where 20DMA Slope crossed ABOVE 200DMA Slope within the last N days.
    Condition:
    - At some point in window: Slope20 > Slope200 AND Prev_Slope20 <= Prev_Slope200
    - Filter: Current 20DMA Slope > min_slope
    """
    crossover_tickers = []
    
    # Filter candidates: 20DMA Slope must be > 200DMA Slope NOW (or recently)
    # Optimization: Only check stocks where latest 20DMA Slope > latest 200DMA Slope?
    # No, crossover could have happened 2 days ago and they are still above.
    # But if they are currently BELOW, then the crossover is invalid/reversed.
    # So yes, Current Slope 20 > Current Slope 200 is a prereq for a "sustained" crossover.
    
    candidates = latest_df[
        (latest_df['20DMA_SLOPE'] > latest_df['200DMA_SLOPE']) &
        (latest_df['20DMA_SLOPE'] > min_20dma_slope)
    ]['Ticker'].tolist()
    
    for ticker in candidates:
        stock_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        # We need lookback_days + 1 records to check crossover
        if len(stock_data) < lookback_days + 2:
            continue
            
        recent_data = stock_data.tail(lookback_days + 1).reset_index(drop=True)
        
        # Check for crossover event in recent history
        for i in range(1, len(recent_data)):
            # Current day i
            curr_20 = recent_data.loc[i, '20DMA_SLOPE']
            curr_200 = recent_data.loc[i, '200DMA_SLOPE']
            
            # Prev day i-1
            prev_20 = recent_data.loc[i-1, '20DMA_SLOPE']
            prev_200 = recent_data.loc[i-1, '200DMA_SLOPE']
            
            if (curr_20 > curr_200) and (prev_20 <= prev_200):
                crossover_tickers.append(ticker)
                break
                
    return crossover_tickers, f"Slope Crossover (Last {lookback_days} days)"

def get_strong_adx_tickers(latest_df, full_df, adx_threshold=25, buy_side_only=True, crossover_only=False):
    """
    Find tickers with ADX > threshold indicating a strong trend.
    If buy_side_only is True, filters for +DI > -DI (Bullish trends only).
    If crossover_only is True, filters for Fresh Bullish Crossover (+DI > -DI today AND +DI <= -DI yesterday).
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
            
            if adx_data.empty or len(adx_data) < 2:
                continue
                
            latest_adx = adx_data.iloc[-1]
            prev_adx = adx_data.iloc[-2]
            
            if latest_adx['ADX'] > adx_threshold:
                # Check Direction if required
                if crossover_only:
                    # Fresh Bullish Crossover: Today Bullish AND Yesterday Bearish/Neutral
                    is_bullish_today = latest_adx['+DI'] > latest_adx['-DI']
                    is_bullish_yesterday = prev_adx['+DI'] > prev_adx['-DI']
                    
                    if is_bullish_today and not is_bullish_yesterday:
                        tickers.append(ticker)
                        
                elif buy_side_only:
                    if latest_adx['+DI'] > latest_adx['-DI']:
                         tickers.append(ticker)
                else:
                    tickers.append(ticker)
                
        except Exception as e:
            continue
            
    return tickers, "Strong ADX"

def search_tickers(latest_df, search_query):
    if not search_query or len(search_query) < 2:
        return [], "Search"
    
    query = search_query.upper()
    mask = latest_df['Ticker'].str.contains(query, case=False, na=False)
    return latest_df[mask]['Ticker'].tolist(), "Search"

def get_heikin_ashi_turnover_tickers(latest_df, full_df, min_angle=0):
    """
    Identify stocks where Heikin Ashi candle turned from Red to Green.
    Red: HA_Close < HA_Open
    Green: HA_Close > HA_Open
    """
    turnover_tickers = []
    
    candidates = latest_df[latest_df['200DMA_SLOPE'] > min_angle]['Ticker'].tolist()
    
    for ticker in candidates:
        stock_df = full_df[full_df['Ticker'] == ticker].sort_values('Date')
        
        if len(stock_df) < 2:
            continue
            
        recent = stock_df.tail(2).reset_index(drop=True)
        
        if 'HA_Open' not in recent.columns or 'HA_Close' not in recent.columns:
            continue
            
        prev_ha_open = recent.loc[0, 'HA_Open']
        prev_ha_close = recent.loc[0, 'HA_Close']
        curr_ha_open = recent.loc[1, 'HA_Open']
        curr_ha_close = recent.loc[1, 'HA_Close']
        
        if (prev_ha_close < prev_ha_open) and (curr_ha_close > curr_ha_open):
            turnover_tickers.append(ticker)
            
    return turnover_tickers, "Heikin Ashi Turn"

def get_heikin_ashi_potential_turn_tickers(latest_df, full_df, min_angle=0):
    """
    Identify stocks where Heikin Ashi red candle bodies are shrinking,
    suggesting weakening bearish momentum and a potential turn to green.
    Criteria:
    - Last 2 candles are both red (HA_Close < HA_Open)
    - Current red body is smaller than previous red body (shrinking bears)
    """
    potential_tickers = []
    
    candidates = latest_df[latest_df['200DMA_SLOPE'] > min_angle]['Ticker'].tolist()
    
    for ticker in candidates:
        stock_df = full_df[full_df['Ticker'] == ticker].sort_values('Date')
        
        if len(stock_df) < 3:
            continue
            
        recent = stock_df.tail(3).reset_index(drop=True)
        
        if 'HA_Open' not in recent.columns or 'HA_Close' not in recent.columns:
            continue
        
        prev2_ha_open = recent.loc[0, 'HA_Open']
        prev2_ha_close = recent.loc[0, 'HA_Close']
        prev_ha_open = recent.loc[1, 'HA_Open']
        prev_ha_close = recent.loc[1, 'HA_Close']
        curr_ha_open = recent.loc[2, 'HA_Open']
        curr_ha_close = recent.loc[2, 'HA_Close']
        
        # Current candle must still be red
        if curr_ha_close >= curr_ha_open:
            continue
        
        # Previous candle must also be red
        if prev_ha_close >= prev_ha_open:
            continue
        
        curr_body = curr_ha_open - curr_ha_close  # positive since red
        prev_body = prev_ha_open - prev_ha_close  # positive since red
        
        # Shrinking red body = weakening bears
        if curr_body < prev_body:
            potential_tickers.append(ticker)
            
    return potential_tickers, "Heikin Ashi Potential Turn"

def check_10_ema_eligibility(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Check if a stock passes the 10 EMA Institutional Trend strategy.
    """
    if len(df) < lookback + 10:
        return False
        
    import pandas_ta as ta
    df = df.copy()
    
    # Calculate 10 EMA
    df.ta.ema(length=10, append=True)
    ema_col = 'EMA_10'
    
    if ema_col not in df.columns:
        return False
        
    recent_data = df.iloc[-lookback:].copy()
    
    # Condition A: Clear Upward Trend
    recent_data['ema_slope'] = recent_data[ema_col].diff()
    ema_rising_percentage = (recent_data['ema_slope'] > 0).mean()
    if ema_rising_percentage < 0.60 or recent_data.iloc[-1][ema_col] <= recent_data.iloc[0][ema_col]:
        return False
        
    # Condition B: EMA Cutting Price
    is_cutting = (
        ((recent_data['open'] > recent_data[ema_col]) & (recent_data['close'] < recent_data[ema_col])) | 
        ((recent_data['open'] < recent_data[ema_col]) & (recent_data['close'] > recent_data[ema_col]))
    )
    cutting_percentage = is_cutting.mean()
    if cutting_percentage > 0.30:
        return False
        
    # Condition C: Respecting the Control Line
    price_above_ema_percentage = (recent_data['close'] > recent_data[ema_col]).mean()
    if price_above_ema_percentage < 0.60:
        return False

    # Condition D: Random Spikes / Illiquidity Filter
    median_vol = recent_data['volume'].median()
    if median_vol < 100000:
        return False
    if recent_data['volume'].max() > (median_vol * 10):
        return False
        
    # Entry Confirmation Rules
    setup_window = recent_data.iloc[-5:]
    trigger_candle = recent_data.iloc[-1]
    
    # Condition E: Pullback to EMA
    touched_ema = (setup_window['low'] <= setup_window[ema_col] * 1.005).any()
    broke_below = (setup_window['close'] < setup_window[ema_col] * 0.99).any()
    
    if not touched_ema or broke_below:
        return False
        
    # Condition F: Volume Contraction on Pullback
    up_candles = recent_data[recent_data['close'] > recent_data['open']]
    down_candles = recent_data[recent_data['close'] < recent_data['open']]
    
    if len(up_candles) == 0 or len(down_candles) == 0:
        return False
        
    avg_up_volume = up_candles['volume'].mean()
    avg_down_volume = down_candles['volume'].mean()
    
    if avg_down_volume > avg_up_volume:
        return False
        
    recent_down_candles = setup_window[setup_window['close'] < setup_window['open']]
    if not recent_down_candles.empty:
        if (recent_down_candles['volume'] > avg_up_volume * 1.5).any():
            return False
            
    # Condition G: Trigger Candle
    is_bullish = trigger_candle['close'] > trigger_candle['open']
    close_above_ema = trigger_candle['close'] > trigger_candle[ema_col]
    
    if not is_bullish or not close_above_ema:
        return False
        
    return True

def get_10_ema_strategy_tickers(latest_df, full_df, lookback=20, min_angle=0):
    """
    Find tickers that meet the 10 EMA Institutional Trend criteria.
    """
    eligible_tickers = []
    
    # Filter candidates by 200DMA angle to speed up
    candidates = latest_df[latest_df['200DMA_SLOPE'] > min_angle]['Ticker'].tolist()
    
    for ticker in candidates:
        stock_df = full_df[full_df['Ticker'] == ticker].sort_values('Date').tail(100)
        
        # Rename columns to lowercase for pandas_ta / our function expectations
        stock_df = stock_df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        if check_10_ema_eligibility(stock_df, lookback):
            eligible_tickers.append(ticker)
            
    return eligible_tickers, "10 EMA Strategy"
