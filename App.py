import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import time
import subprocess
import sys
import streamlit.components.v1 as components 
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import json
import analysis_logic as al
import metrics as mt

# Page configuration
st.set_page_config(
    page_title="NSE Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
DATA_FILE = "nse_stock_data_with_metrics_v2.csv"
DOCS_FILE = "app_documentation.json"
FRESHNESS_HOURS = 20
CSS_FILE = "style.css"

def load_css(is_dark_mode=True):
    if is_dark_mode:
        with open(CSS_FILE) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)




def check_and_update_data():
    """Checks if data is fresh, otherwise runs the update script."""
    should_update = False
    msg = ""

    if not os.path.exists(DATA_FILE):
        should_update = True
        msg = "Data file not found. Initializing data fetch..."
    else:
        mod_time = os.path.getmtime(DATA_FILE)
        age_hours = (time.time() - mod_time) / 3600
        if age_hours > FRESHNESS_HOURS:
            should_update = True
            msg = f"Data is {age_hours:.1f} hours old (Limit: {FRESHNESS_HOURS}h). Updating..."
    
    if should_update:
        with st.spinner(msg):
            try:
                # Run add_metrics.py using the same python environment
                subprocess.run([sys.executable, "add_metrics.py"], check=True)
                st.success("Data updated successfully!")
                # Clear cache to force reload of new data
                load_data.clear()
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to update data: {e}")
            except Exception as e:
                st.error(f"An error occurred during update: {e}")

@st.cache_data
def load_data():
    """Load and preprocess the stock data."""
    try:
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by Ticker and Date
        df = df.sort_values(by=['Ticker', 'Date'])
        
        # Calculate Daily Returns for Beta
        df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()

        # Calculate Previous Day's 20DMA and 200DMA for Crossover Logic
        df['Prev_20DMA'] = df.groupby('Ticker')['20DMA'].shift(1)
        df['Prev_200DMA'] = df.groupby('Ticker')['200DMA'].shift(1)
        # Calculate Previous Day's DMA Slopes for Bottoming Logic (3, 20, 200)
        for dma in [3, 20, 200]:
            if f'{dma}DMA_SLOPE' in df.columns:
                df[f'Prev_{dma}DMA_SLOPE'] = df.groupby('Ticker')[f'{dma}DMA_SLOPE'].shift(1)

        # Calculate 20-Day Volume SMA for Volume Shockers
        df['Volume_20SMA'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=20).mean())
        
        # Ensure we have the necessary angle columns if they aren't explicitly loaded (though csv should have them)
        # Based on add_metrics.py: 200DMA_LINE_ANGLE, RSI_14_angle
        
        # Load Benchmark Data
        benchmark_df = df[df['Ticker'] == 'SBINEQWETF.NS'].copy()
        if benchmark_df.empty and 'SBINEQWETF' in df['Ticker'].unique():
             benchmark_df = df[df['Ticker'] == 'SBINEQWETF'].copy()
        
        return df, benchmark_df
    except FileNotFoundError:
        st.error(f"File {DATA_FILE} not found. Please run the data fetcher first.")
        return pd.DataFrame(), pd.DataFrame()

def load_documentation():
    """Load documentation from JSON file."""
    try:
        with open(DOCS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback empty dict if file missing, to prevent crash
        return {}

# --- 3. Calculation Functions ---
# Moved to metrics.py
    
def plot_charts(df, ticker, days=180, candle_type="Heikin Ashi", show_bollinger=True, show_volume_profile=False, show_slope=False, show_macd=False, show_adx=False, show_rs=False, benchmark_df=None, is_dark_mode=True):
    """Render interactive Plotly charts for a specific ticker."""
    stock_df = df[df['Ticker'] == ticker].copy()
    stock_df = stock_df.sort_values('Date')
    
    # Filter for the selected timeframe
    stock_df = stock_df.tail(days)

    # Calculate Heikin Ashi if selected
    if candle_type == "Heikin Ashi":
        chart_df = mt.calculate_heikin_ashi(stock_df)
    else:
        chart_df = stock_df

    # Calculate Bollinger Bands if requested (using original Close prices for accuracy)
    if show_bollinger:
        bb_data = mt.calculate_bollinger_bands(stock_df)
        
    # Calculate Volume Profile if requested
    if show_volume_profile:
        vp_data = mt.calculate_volume_profile(stock_df)

    # Calculate MACD if requested
    if show_macd:
        macd_data = mt.calculate_macd(stock_df)

    # Calculate ADX if requested
    if show_adx:
        adx_data = mt.calculate_adx(stock_df)

    # Determine Row Layout
    # Row 1: Price
    # Row 2: Volume
    # Row 3: RSI
    # Row 4: Slope History (Optional)
    # Row 5: MACD (Optional)
    # Row 6: RS (Optional)
    # Row 7: ADX (Optional)
    
    rows = 3
    # Adjusted heights: Main (0.6), Volume (0.1), RSI (0.15)
    row_heights = [0.6, 0.10, 0.15]
    specs = [[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    
    slope_row = None
    macd_row = None
    rs_row = None
    adx_row = None
    
    if show_slope:
        rows += 1
        slope_row = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])
        
    if show_macd:
        rows += 1
        macd_row = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])
        
    if show_rs:
        rows += 1
        rs_row = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    if show_adx:
        rows += 1
        adx_row = rows
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    # Normalize row heights logic is handled by Plotly usually, but let's just pass relative weights
    # Or just use the defaults for now which are equal. 
    # Better: explicitly define height based on count.
    
    total_height = 700 + (200 * (rows - 3)) 

    fig = make_subplots(
        rows=rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=row_heights, 
        specs=specs
    )

    # 1. Price Chart with moving averages
    
    # Volume Profile (Horizontal Bars) - Add FIRST so it's behind candles? Or use opacity.
    # We use a secondary x-axis for this.
    if show_volume_profile:
        fig.add_trace(go.Bar(
            y=vp_data['Price'],
            x=vp_data['Volume'],
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(200, 200, 200, 0.5)',
            xaxis='x2', # Use secondary x-axis
            showlegend=False,
            hoverinfo='none' # Reduce clutter
        ), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=chart_df['Date'],
        open=chart_df['Open'], high=chart_df['High'],
        low=chart_df['Low'], close=chart_df['Close'],
        name=f'Price ({candle_type})'
    ), row=1, col=1)

    # Add Bollinger Bands
    if show_bollinger:
        fig.add_trace(go.Scatter(
            x=bb_data['Date'], y=bb_data['Upper'],
            line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
            name='Upper BB', showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=bb_data['Date'], y=bb_data['Lower'],
            line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
            fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)',
            name='Lower BB', showlegend=False
        ), row=1, col=1)

    # Add 20 DMA (using original data for accuracy)
    if '20DMA' in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df['Date'], y=stock_df['20DMA'],
            line=dict(color='blue', width=1), name='20 DMA'
        ), row=1, col=1)

    # Add 200 DMA
    if '200DMA' in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df['Date'], y=stock_df['200DMA'],
            line=dict(color='red', width=2), name='200 DMA'
        ), row=1, col=1)
        
    # Add 20 VWMA if available
    if '20VWMA' in stock_df.columns:
         fig.add_trace(go.Scatter(
            x=stock_df['Date'], y=stock_df['20VWMA'],
            line=dict(color='orange', width=1, dash='dot'), name='20 VWMA'
        ), row=1, col=1)

    # Determine colors for volume bars based on price direction (Close >= Open)
    volume_colors = ['#00e676' if c >= o else '#ff5252' for c, o in zip(stock_df['Close'], stock_df['Open'])]

    # 2. Volume Chart
    # Color Volume based on Price Action (Close >= Open -> Green, else Red)
    # We use stock_df (original data) for this logic to reflect true market pressure
    volume_colors = ['#00e676' if c >= o else '#ff1744' for c, o in zip(stock_df['Close'], stock_df['Open'])]

    fig.add_trace(go.Bar(
        x=stock_df['Date'], y=stock_df['Volume'],
        name='Volume', marker_color=volume_colors
    ), row=2, col=1)

    # 3. RSI Chart
    if 'RSI_14' in stock_df.columns:
        fig.add_trace(go.Scatter(
            x=stock_df['Date'], y=stock_df['RSI_14'],
            line=dict(color='purple', width=2), name='RSI (14)'
        ), row=3, col=1)
        
        # Add RSI Levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Remove Supertrend plotting code




    # 4. Slope History Chart
    if show_slope and slope_row:
        if '200DMA_SLOPE' in stock_df.columns:
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['200DMA_SLOPE'],
                line=dict(color='yellow', width=2), name='200 DMA Slope'
            ), row=slope_row, col=1)
            
        if '20DMA_SLOPE' in stock_df.columns:
             fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['20DMA_SLOPE'],
                line=dict(color='cyan', width=1, dash='dot'), name='20 DMA Slope'
            ), row=slope_row, col=1)
            
        fig.add_hline(y=0, line_color="gray", row=slope_row, col=1)

    # 5. MACD Chart
    if show_macd and macd_row:
        # Histogram colors
        hist_colors = ['#00e676' if v >= 0 else '#ff1744' for v in macd_data['Hist']]
        
        fig.add_trace(go.Bar(
            x=macd_data['Date'], y=macd_data['Hist'],
            marker_color=hist_colors, name='MACD Hist'
        ), row=macd_row, col=1)
        
        fig.add_trace(go.Scatter(
            x=macd_data['Date'], y=macd_data['MACD'],
            line=dict(color='white', width=1), name='MACD'
        ), row=macd_row, col=1)
        
        fig.add_trace(go.Scatter(
            x=macd_data['Date'], y=macd_data['Signal'],
            line=dict(color='orange', width=1), name='Signal'
        ), row=macd_row, col=1)

    # Layout updates
    layout_update = dict(
        title=f"{ticker} Technical Analysis ({candle_type})",
        xaxis_rangeslider_visible=False,
        height=total_height,
        showlegend=True,
        template="plotly_dark" if is_dark_mode else "plotly_white",
        plot_bgcolor='#0e1117' if is_dark_mode else '#ffffff',
        paper_bgcolor='#0e1117' if is_dark_mode else '#ffffff',
        font=dict(color='#fafafa' if is_dark_mode else '#000000')
    )
    
    if show_volume_profile:
        # Configure secondary x-axis for Volume Profile
        # We want it to be on top or bottom? Or just overlay?
        # 'overlaying="x"' means it shares the same y-axis area but has its own x-axis.
        # 'side="top"' puts the axis labels on top (we can hide them).
        # We reverse it or set range so bars appear on the right or left?
        # Usually VP is on the right or left.
        # Let's put it on the right side, growing leftwards? Or left side growing rightwards?
        # Let's try left side growing rightwards, but with a range that makes them take up only ~20-30% of the width.
        
        max_vol = vp_data['Volume'].max()
        # Set range from 0 to max_vol * 4 (so bars take 1/4th of width)
        layout_update['xaxis2'] = dict(
            overlaying='x', 
            side='top', 
            showgrid=False, 
            range=[0, max_vol * 5], 
            visible=False # Hide the axis itself
        )

    fig.update_layout(**layout_update)
    
    # Customize axes to reduce grid clutter
    # Price Chart (Row 1): Subtle Grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=1, col=1)
    
    # Volume Chart (Row 2): No Grid (as requested)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=2, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    
    # RSI Chart (Row 3): Subtle Grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=3, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=3, col=1)
    
    # Slope Chart Grid
    if show_slope and slope_row:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=slope_row, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', title="Slope (°)", row=slope_row, col=1)

    # MACD Chart Grid
    if show_macd and macd_row:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', row=macd_row, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', title="MACD", row=macd_row, col=1)

    # 6. Relative Strength Chart
    if show_rs and rs_row and benchmark_df is not None and not benchmark_df.empty:
        rs_data = mt.calculate_relative_strength(stock_df, benchmark_df)
        fig.add_trace(go.Scatter(
            x=rs_data['Date'], y=rs_data['RS_Ratio'],
            line=dict(color='orange', width=2), name='RS Ratio'
        ), row=rs_row, col=1)
        
        # Add MA for RS
        fig.add_trace(go.Scatter(
            x=rs_data['Date'], y=rs_data['RS_MA'],
            line=dict(color='gray', width=1, dash='dot'), name='RS MA(50)'
        ), row=rs_row, col=1)
        fig.update_yaxes(title="RS Ratio", row=rs_row, col=1)

    # 7. ADX Chart
    if show_adx and adx_row:
        # ADX Line (White/Black based on theme?) - Using distinct color like Gold or White
        adx_color = 'white' if is_dark_mode else 'black'
        
        fig.add_trace(go.Scatter(
            x=adx_data['Date'], y=adx_data['ADX'],
            line=dict(color=adx_color, width=2), name='ADX'
        ), row=adx_row, col=1)
        
        # +DI (Green)
        fig.add_trace(go.Scatter(
            x=adx_data['Date'], y=adx_data['+DI'],
            line=dict(color='#00e676', width=1), name='+DI'
        ), row=adx_row, col=1)
        
        # -DI (Red)
        fig.add_trace(go.Scatter(
            x=adx_data['Date'], y=adx_data['-DI'],
            line=dict(color='#ff1744', width=1), name='-DI'
        ), row=adx_row, col=1)
        
        # Threshold Reference Line (25)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=adx_row, col=1)
        fig.update_yaxes(title="ADX", row=adx_row, col=1)

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 NSE Stock Analysis Dashboard")
    
    # Check for data freshness before loading
    check_and_update_data()
    
    with st.spinner("Loading data..."):
        df, benchmark_df = load_data()
        docs = load_documentation()
        
    if df.empty:
        return

    with st.spinner("Calculating metrics..."):
        # Wrap the cached calculation
        @st.cache_data
        def get_cached_beta(data):
            return mt.calculate_beta_metrics(data)
            
        beta_map = get_cached_beta(df)

    # Calculate Volume Metrics (On-the-fly)
    # We need 20-day average volume. 
    # Since we have the full df, we can calculate it.
    # Note: df is sorted by Ticker and Date.
    df['20DayAvgVolume'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
    
    # Get the latest data for each ticker for filtering logic
    # We essentially want to check the criteria based on the *latest* available date for each stock.
    latest_df = df.sort_values(by=['Ticker', 'Date']).groupby('Ticker').tail(1).copy()
    
    # Map Beta to latest_df
    latest_df['Beta'] = latest_df['Ticker'].map(beta_map)
    
    # Calculate Volume Ratio for latest data
    latest_df['VolumeRatio'] = latest_df['Volume'] / latest_df['20DayAvgVolume']
    
    # --- Sidebar Controls ---
    st.sidebar.header("Chart Settings")
    
    # Theme Toggle
    is_dark_mode = st.sidebar.checkbox("Dark Mode", value=True)
    load_css(is_dark_mode)

    timeframe_options = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "Max": 5000}
    selected_timeframe = st.sidebar.selectbox("Timeframe", list(timeframe_options.keys()), index=1)
    timeframe_days = timeframe_options[selected_timeframe]

    candle_type = st.sidebar.radio("Candle Type", ["Heikin Ashi", "Normal"], index=0)
    show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_volume_profile = st.sidebar.checkbox("Show Volume Profile", value=False)
    
    st.sidebar.markdown("---")
    show_slope = st.sidebar.checkbox("Show Slope History", value=False)
    show_macd = st.sidebar.checkbox("Show MACD", value=True)
    show_adx = st.sidebar.checkbox("Show ADX", value=True)
    show_rs = st.sidebar.checkbox("Show RS (vs Nifty ETF)", value=False)

    # --- Top Navigation ---
    tab_options = [
        "Low RSI", 
        "Low Beta", 
        "Narrow Band", 
        "Divergence Slope",
        "Slope Difference",
        "High DMA Angle",
        "Actual Crossover",
        "Potential Crossover",
        "Volume Shockers",
        "DMA Bottoming",
        "MACD Crossover",
        "Relative Strength",
        "Strong ADX"
    ]
    
    selected_tab = st.radio("Select Analysis Mode", tab_options, horizontal=True, label_visibility="collapsed")
    st.markdown("---")


    st.sidebar.header("Filter Settings")
    
    # Initialize variables with defaults (to avoid NameError)
    t1_rsi, t1_angle = 30, 5
    t2_beta_pct, t2_angle = 0.25, 5
    t3_band_pct, t3_days, t3_angle = 0.05, 10, 5
    t4_angle = 5
    t5_top_n, t5_angle = 10, 5
    crossover_angle, proximity_pct = 5, 2.0
    vol_shock_threshold, min_volume, t8_angle = 2.0, 10000, 5
    t9_min_angle, t9_max_angle, t9_prev_max = -2.0, 10.0, 0.0

    # Conditional Sidebar Controls
    if selected_tab == "Low RSI":
        st.sidebar.subheader("Low RSI Settings")
        t1_rsi = st.sidebar.slider("RSI Threshold", 0, 100, 30, key="t1_rsi")
        t1_angle = st.sidebar.number_input("200 DMA Angle >", value=5, key="t1_angle")
        
    elif selected_tab == "Low Beta":
        st.sidebar.subheader("Low Beta Settings")
        t2_beta_pct = st.sidebar.slider("Beta Percentile (Bottom %)", 0.05, 0.5, 0.25, 0.05, key="t2_beta")
        t2_angle = st.sidebar.number_input("200 DMA Angle >", value=5, key="t2_angle") 

    elif selected_tab == "Narrow Band":
        st.sidebar.subheader("Narrow Band Settings")
        t3_band_pct = st.sidebar.slider("Price Band %", 0.01, 0.20, 0.05, 0.01, key="t3_band")
        t3_days = st.sidebar.slider("Lookback Days", 5, 30, 10, key="t3_days")
        t3_angle = st.sidebar.number_input("200 DMA Angle >", value=5, key="t3_angle")

    elif selected_tab == "Divergence Slope":
        st.sidebar.subheader("Divergence Slope Settings")
        t4_angle = st.sidebar.number_input("200 DMA Slope >", value=5, key="t4_angle")

    elif selected_tab == "High DMA Angle":
        st.sidebar.subheader("DMA Angle Settings")
        selected_dma = st.sidebar.selectbox("Select DMA", [3, 20, 200], index=0)
        t5_top_n = st.sidebar.number_input("Top N Stocks", value=10, min_value=5, max_value=50, key="t5_top_n")
        t5_angle = st.sidebar.number_input("200 DMA Slope >", value=5, key="t5_angle")

    elif selected_tab == "Slope Difference":
        st.sidebar.subheader("Slope Difference Settings")
        t6_top_n = st.sidebar.number_input("Top N Stocks", value=20, min_value=5, max_value=50, key="t6_top_n")
        t6_min_diff = st.sidebar.number_input("Min Slope Difference", value=0.0, step=0.5, key="t6_min_diff")
        t6_min_200_slope = st.sidebar.number_input("200 DMA Slope >", value=0.0, step=0.5, key="t6_min_200_slope")
        
    elif selected_tab == "Actual Crossover" or selected_tab == "Potential Crossover":
        st.sidebar.subheader("Crossover Settings")
        crossover_angle = st.sidebar.number_input("200 DMA Slope >", value=5, key="crossover_angle")
        if selected_tab == "Potential Crossover":
            proximity_pct = st.sidebar.slider("Potential Proximity %", 0.1, 5.0, 2.0, 0.1, key="proximity_pct")
            
    elif selected_tab == "Volume Shockers":
        st.sidebar.subheader("Volume Shockers Settings")
        vol_shock_threshold = st.sidebar.slider("Volume Ratio (> x times avg)", 1.0, 10.0, 2.0, 0.1, key="vol_shock_threshold")
        min_volume = st.sidebar.number_input("Min Average Volume", value=10000, step=10000, key="min_volume")
        t8_angle = st.sidebar.number_input("200 DMA Slope >", value=5, key="t8_angle")

    elif selected_tab == "DMA Bottoming":
        st.sidebar.subheader("DMA Bottoming Settings")
        selected_dma_bot = st.sidebar.selectbox("Select DMA", [20, 3, 200], index=0, key="dma_bot")
        t9_min_angle = st.sidebar.number_input(f"Min Current {selected_dma_bot}DMA Angle", value=-2.0, step=0.5, key=f"t9_min_{selected_dma_bot}")
        t9_max_angle = st.sidebar.number_input(f"Max Current {selected_dma_bot}DMA Angle", value=10.0, step=0.5, key=f"t9_max_{selected_dma_bot}")
        t9_prev_max = st.sidebar.number_input(f"Max Previous {selected_dma_bot}DMA Angle", value=0.0, step=0.5, key=f"t9_prev_{selected_dma_bot}")
        
    elif selected_tab == "MACD Crossover":
        st.sidebar.subheader("MACD Crossover Settings")
        crossover_angle = st.sidebar.number_input("200 DMA Angle >", value=5, key="macd_angle")
        macd_lookback = st.sidebar.slider("Lookback Days (Signal)", 1, 10, 3, key="macd_lookback")

    elif selected_tab == "Relative Strength":
        st.sidebar.subheader("RS Settings")
        rs_slope_min = st.sidebar.number_input("Min RS Slope >", value=0, key="rs_slope")
        
    elif selected_tab == "Strong ADX":
        st.sidebar.subheader("ADX Settings")
        adx_threshold = st.sidebar.number_input("Min ADX", value=25, key="adx_threshold")

    # --- Filter Logic and Display ---
    
    # helper to filter latest_df
    def get_display_data(tickers, include_vol_metrics=False):
        # Standardized columns for all tabs
        # We include all key metrics to provide a consistent view
        cols = [
            'Ticker', 'Close', 'RSI_14', 'RSI_14_SLOPE', 
            'Beta', 'Volume', 'VolumeRatio', 
            '3DMA_SLOPE', '20DMA_SLOPE', '200DMA_SLOPE',
            '20DMA', '200DMA'
        ]
        # Remove potential duplicates just in case
        cols = list(dict.fromkeys(cols))
            
        # Filter rows
        subset = latest_df[latest_df['Ticker'].isin(tickers)][cols].copy()
        
        # Rename for display
        rename_map = {
            'Ticker': 'Ticker', 'Close': 'Price', 
            'RSI_14': 'RSI', 'RSI_14_SLOPE': 'RSI Slope',
            'Beta': 'Beta',
            'Volume': 'Volume', 'VolumeRatio': 'Vol Ratio',
            '3DMA_SLOPE': '3DMA Slope',
            '20DMA_SLOPE': '20DMA Slope',
            '200DMA_SLOPE': '200DMA Slope',
            '20DMA': '20 DMA (Price)', '200DMA': '200 DMA (Price)'
        }
            
        subset = subset.rename(columns=rename_map)
        # Round appropriate columns
        subset = subset.round(2)
        return subset

    def render_tab_content(data_df, description, key_prefix, documentation=None):
        st.markdown(f"**{description}**")
        
        if documentation:
            with st.expander("ℹ️ Feature Documentation"):
                st.markdown(documentation)
        st.markdown(f"Found {len(data_df)} stocks.")
        
        if data_df.empty:
            st.info("No stocks matched this criteria.")
            return

        # Configure AgGrid
        gb = GridOptionsBuilder.from_dataframe(data_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20) # Optional pagination
        gb.configure_side_bar() # Add sidebar for columns/filters
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False, resizable=True, filterable=True, sortable=True)
        
        # Selection
        gb.configure_selection('single', use_checkbox=True, groupSelectsChildren=True) 
        
        # Enable Copy to Clipboard (Enterprise feature mostly, but standard copy works in community if configured)
        # Actually standard browser copy works. enabling range selection helps.
        gb.configure_grid_options(enableRangeSelection=True) 
        
        # Conditional Formatting (Simplified for AgGrid via CellStyle - easier to just rely on Dataframe styling if passed? 
        # AgGrid in Streamlit doesn't easily inherit pandas styler. We need JS functions for cell styles.
        # For this iteration, we might lose the colorful gradients unless we implement JS. 
        # Given the "Improve grid" request, function > aesthetics here?
        # User asked for "standardise grid columns" and "copy/export".
        # I will prioritize those. Gradients are nice but AgGrid is complex with them.
        # I'll stick to basic grid for now.
        
        # Button Layout: Push to right side (Icon buttons)
        col_spacer, col_dl, col_copy = st.columns([15, 1, 1])

        with col_dl:
             csv = data_df.to_csv(index=False).encode('utf-8')
             st.download_button(
                 label="📥",
                 data=csv,
                 file_name=f"{key_prefix}_data.csv",
                 mime="text/csv",
                 key=f'download-csv-{key_prefix}',
                 help="Download CSV"
             )

        with col_copy:
            # Copy to Clipboard Button (Icon Style)
            clipboard_text = data_df.to_csv(index=False, sep='\t') 
            clipboard_text_js = clipboard_text.replace('`', '').replace('"', '\\"').replace('\n', '\\n')
            
            # JS to copy text - styled as a small icon button
            copy_js = f"""
            <div style="text-align: center;">
                <button onclick="copyToClipboard()" title="Copy to Clipboard" style="
                    background-color: #262730; 
                    color: white; 
                    border: 1px solid #464b5d; 
                    padding: 0.25rem 0.5rem; 
                    border-radius: 0.25rem; 
                    cursor: pointer;
                    font-size: 1.2rem;
                    line-height: 1;
                    width: auto;
                    transition: all 0.2s;
                " onmouseover="this.style.borderColor='#fafafa';this.style.backgroundColor='#464b5d'" 
                  onmouseout="this.style.borderColor='#464b5d';this.style.backgroundColor='#262730'">
                    📋
                </button>
                <div id="copy_msg_{key_prefix}" style="
                    color: #00e676; 
                    font-size: 0.7rem; 
                    opacity: 0; 
                    transition: opacity 0.5s; 
                    margin-top: 2px;
                    text-align: center;
                    white-space: nowrap;
                    margin-left: -10px;
                ">Copied!</div>
            </div>
            <script>
                function copyToClipboard() {{
                    const text = `{clipboard_text_js}`;
                    navigator.clipboard.writeText(text).then(function() {{
                        const msg = document.getElementById("copy_msg_{key_prefix}");
                        msg.style.opacity = 1;
                        setTimeout(function() {{ msg.style.opacity = 0; }}, 2000);
                    }}, function(err) {{
                        console.error('Async: Could not copy text: ', err);
                    }});
                }}
            </script>
            """
            st.components.v1.html(copy_js, height=50)

        gridOptions = gb.build()
        
        grid_response = AgGrid(
            data_df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT', 
            update_mode='SELECTION_CHANGED', # Only update when selection changes
            fit_columns_on_grid_load=False,
            theme='balham', # 'streamlit', 'alpine', 'balham', 'material'
            enable_enterprise_modules=False,
            height=400, 
            width='100%',
            reload_data=False,
            key=f"grid_{key_prefix}"
        )

        selected = grid_response['selected_rows']
        
        # Robust selection check to avoid "ambiguous truth value" error
        has_selection = False
        if isinstance(selected, list):
            has_selection = len(selected) > 0
        elif isinstance(selected, pd.DataFrame):
            has_selection = not selected.empty
            
        if has_selection: 
            # AgGrid returns a list of dictionaries/rows or a DataFrame
            # Since we selected single, it's a list of 1 dict or 1 row DF
            if isinstance(selected, pd.DataFrame):
                 selected_row = selected.iloc[0]
                 selected_ticker = selected_row['Ticker']
            elif isinstance(selected, list):
                 selected_ticker = selected[0]['Ticker']
                 selected_row = selected[0] # Dict
            else:
                 return

            st.markdown("---")
            st.subheader(f"Analysis for {selected_ticker}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            # Access keys safely (dict or series)
            col1.metric("Close Price", f"{selected_row['Price']:.2f}")
            col2.metric("RSI (14)", f"{selected_row['RSI']:.2f}")
            col3.metric("200 DMA Slope", f"{selected_row['200DMA Slope']:.2f}°")
            col4.metric("3 DMA Slope", f"{selected_row['3DMA Slope']:.2f}°")
            col5.metric("Beta", f"{selected_row['Beta']:.2f}")
            
            plot_charts(df, selected_ticker, timeframe_days, candle_type, show_bollinger, show_volume_profile, show_slope, show_macd, show_adx, show_rs, benchmark_df, is_dark_mode)
        else:
            st.info("👆 Select a row (checkmark) to view the chart.")

    # Render Active Tab Content Only
    if selected_tab == "Low RSI":
        tickers, doc_key = al.get_low_rsi_tickers(latest_df, t1_rsi, t1_angle)
        df_1 = get_display_data(tickers).sort_values(by=['RSI', 'RSI Slope'], ascending=[True, True])
        render_tab_content(df_1, f"RSI < {t1_rsi} and 200 DMA Slope > {t1_angle}°", "tab1", docs.get(doc_key))
        
    elif selected_tab == "Low Beta":
        tickers, doc_key = al.get_low_beta_tickers(latest_df, t2_beta_pct, t2_angle)
        df_2 = get_display_data(tickers).sort_values(by='Beta', ascending=True)
        render_tab_content(df_2, f"Low Beta (Bottom {int(t2_beta_pct*100)}%) and 200 DMA Slope > {t2_angle}°", "tab2", docs.get(doc_key))

    elif selected_tab == "Narrow Band":
        tickers, doc_key = al.get_narrow_band_tickers(latest_df, df, t3_band_pct, t3_days, t3_angle)
        df_3 = get_display_data(tickers)
        render_tab_content(df_3, f"Narrow Price Band ({int(t3_band_pct*100)}% range/{t3_days} days) and 200 DMA Slope > {t3_angle}°", "tab3", docs.get(doc_key))

    elif selected_tab == "Divergence Slope":
        tickers, doc_key = al.get_divergence_slope_tickers(latest_df, t4_angle)
        df_4 = get_display_data(tickers, include_vol_metrics=True).sort_values(by='3DMA Slope', ascending=False)
        render_tab_content(df_4, f"Top 10 High 3DMA Slope (Divergence) with 200 DMA Slope > {t4_angle}°", "tab4", docs.get(doc_key))

    elif selected_tab == "High DMA Angle":
        tickers, doc_key = al.get_high_dma_angle_tickers(latest_df, selected_dma, t5_top_n, t5_angle)
        display_name = f"{selected_dma}DMA Slope"
        df_5 = get_display_data(tickers, include_vol_metrics=True).sort_values(by=display_name, ascending=False)
        render_tab_content(df_5, f"Top {t5_top_n} Stocks by Highest {display_name} (200DMA Slope > {t5_angle}°)", "tab5", docs.get(doc_key))

    elif selected_tab == "Slope Difference":
        tickers, doc_key = al.get_slope_difference_tickers(latest_df, t6_min_diff, t6_min_200_slope, t6_top_n)
        df_diff = get_display_data(tickers)
        df_diff['Slope Diff'] = df_diff['20DMA Slope'] - df_diff['200DMA Slope']
        render_tab_content(df_diff, f"Top {t6_top_n} Stocks by (20DMA Slope - 200DMA Slope) >= {t6_min_diff}", "tab_slope_diff", docs.get(doc_key))

    elif selected_tab == "Actual Crossover":
        tickers, doc_key = al.get_actual_crossover_tickers(latest_df, crossover_angle)
        df_6 = get_display_data(tickers)
        render_tab_content(df_6, f"Actual Bullish Crossover (20DMA crosses 200DMA) + Slope > {crossover_angle}°", "tab6", docs.get(doc_key))

    elif selected_tab == "Potential Crossover":
        tickers, doc_key = al.get_potential_crossover_tickers(latest_df, proximity_pct, crossover_angle)
        df_7 = get_display_data(tickers)
        render_tab_content(df_7, f"Potential Bullish Crossover (Gap < {proximity_pct}%) + Slope > {crossover_angle}°", "tab7", docs.get(doc_key))

    elif selected_tab == "Volume Shockers":
        tickers, doc_key = al.get_volume_shockers_tickers(latest_df, vol_shock_threshold, min_volume, t8_angle)
        df_8 = get_display_data(tickers, include_vol_metrics=True).sort_values(by='Vol Ratio', ascending=False)
        render_tab_content(df_8, f"Volume > {vol_shock_threshold}x Avg AND Avg Vol > {min_volume} AND Slope > {t8_angle}°", "tab8", docs.get(doc_key))

    elif selected_tab == "DMA Bottoming":
        tickers, doc_key = al.get_dma_bottoming_tickers(latest_df, selected_dma_bot, t9_min_angle, t9_max_angle, t9_prev_max)
        display_name_bot = f"{selected_dma_bot}DMA Slope"
        df_9 = get_display_data(tickers, include_vol_metrics=True).sort_values(by=display_name_bot, ascending=True)
        
        # Dynamic format for this one
        doc_10 = docs.get(doc_key, "")
        if doc_10:
             doc_10 = doc_10.format(selected_dma_bot=selected_dma_bot)

        render_tab_content(df_9, f"{selected_dma_bot}DMA Bottoming: Prev Angle <= {t9_prev_max}° → Current [{t9_min_angle}°, {t9_max_angle}°] (Turning Up)", "tab9", doc_10)

    elif selected_tab == "MACD Crossover":
        tickers, doc_key = al.get_macd_crossover_tickers(latest_df, df, crossover_angle, macd_lookback)
        df_10 = get_display_data(tickers)
        render_tab_content(df_10, f"MACD Crossed Above Signal Line (Last {macd_lookback} Days) (+200DMA Slope > {crossover_angle}°)", "tab10", docs.get(doc_key))

    elif selected_tab == "Relative Strength":
        tickers, doc_key = al.get_rs_strong_tickers(latest_df, df, benchmark_df, rs_slope_min)
        df_11 = get_display_data(tickers)
        render_tab_content(df_11, f"Relative Strength > Benchmark (Slope > {rs_slope_min}°)", "tab11", docs.get(doc_key))
        
    elif selected_tab == "Strong ADX":
        tickers, doc_key = al.get_strong_adx_tickers(latest_df, df, adx_threshold)
        df_12 = get_display_data(tickers)
        render_tab_content(df_12, f"ADX > {adx_threshold} (Strong Trend)", "tab12", docs.get(doc_key))

if __name__ == "__main__":
    main()