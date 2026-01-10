import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import time
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="NSE Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
DATA_FILE = "nse_stock_data_with_metrics.csv"
FRESHNESS_HOURS = 20

# Custom CSS for Dark Background and Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Global Text Visibility Force */
    .stMarkdown, .stText, p, label, .stRadio div, .stCheckbox div {
        color: #fafafa !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #464b5d;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stRadio div, 
    [data-testid="stSidebar"] .stCheckbox div {
        color: #ffffff !important;
    }

    /* Inputs in Sidebar */
    [data-testid="stSidebar"] input {
        background-color: #464b5d !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    
    /* Header Styling */
    header[data-testid="stHeader"] {
        background-color: #262730;
        border-bottom: 1px solid #464b5d;
    }
    
    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent;
        color: #fafafa;
        border: 1px solid transparent;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
         background-color: #464b5d;
         color: #00e676 !important;
         border: 1px solid #00e676;
    }
    
    /* Buttons in the grid */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        background-color: #262730;
        color: white;
        border: 1px solid #464b5d;
    }
    .stButton > button:hover {
        background-color: #464b5d;
        border-color: #fafafa;
        color: white;
    }
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #00e676; /* Accent color for metric values */
    }
</style>
""", unsafe_allow_html=True)

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

        # Calculate 20-Day Volume SMA for Volume Shockers
        df['Volume_20SMA'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=20).mean())
        
        # Ensure we have the necessary angle columns if they aren't explicitly loaded (though csv should have them)
        # Based on add_metrics.py: 200DMA_LINE_ANGLE, RSI_14_angle
        
        return df
    except FileNotFoundError:
        st.error(f"File {DATA_FILE} not found. Please run the data fetcher first.")
        return pd.DataFrame()

@st.cache_data
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
    
    # Calculate subsequent HA Opens
    # HA_Open = (Prev_HA_Open + Prev_HA_Close) / 2
    # Note: We use the *calculated* HA values for the previous candle
    
    # However, standard formula often uses: (Prev_HA_Open + Prev_HA_Close) / 2
    # Let's do it iteratively
    
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

def plot_charts(df, ticker, days=180, candle_type="Heikin Ashi", show_bollinger=True, show_volume_profile=False):
    """Render interactive Plotly charts for a specific ticker."""
    stock_df = df[df['Ticker'] == ticker].copy()
    stock_df = stock_df.sort_values('Date')
    
    # Filter for the selected timeframe
    stock_df = stock_df.tail(days)

    # Calculate Heikin Ashi if selected
    if candle_type == "Heikin Ashi":
        chart_df = calculate_heikin_ashi(stock_df)
    else:
        chart_df = stock_df

    # Calculate Bollinger Bands if requested (using original Close prices for accuracy)
    if show_bollinger:
        bb_data = calculate_bollinger_bands(stock_df)
        
    # Calculate Volume Profile if requested
    if show_volume_profile:
        vp_data = calculate_volume_profile(stock_df)

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
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

    # Layout updates
    layout_update = dict(
        title=f"{ticker} Technical Analysis ({candle_type})",
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=True,
        template="plotly_dark",
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa')
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

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 NSE Stock Analysis Dashboard")
    
    # Check for data freshness before loading
    check_and_update_data()
    
    with st.spinner("Loading data..."):
        df = load_data()
        
    if df.empty:
        return

    with st.spinner("Calculating metrics..."):
        beta_map = calculate_beta_metrics(df)

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
    timeframe_options = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "Max": 5000}
    selected_timeframe = st.sidebar.selectbox("Timeframe", list(timeframe_options.keys()), index=1)
    timeframe_days = timeframe_options[selected_timeframe]

    candle_type = st.sidebar.radio("Candle Type", ["Heikin Ashi", "Normal"], index=0)
    show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_volume_profile = st.sidebar.checkbox("Show Volume Profile", value=False)

    st.sidebar.header("Filter Settings")
    
    with st.sidebar.expander("Tab 1: Low RSI", expanded=True):
        t1_rsi = st.slider("RSI Threshold", 0, 100, 30, key="t1_rsi")
        t1_angle = st.number_input("200 DMA Angle >", value=5, key="t1_angle")

    with st.sidebar.expander("Tab 2: Low Beta"):
        t2_beta_pct = st.slider("Beta Percentile (Bottom %)", 0.05, 0.5, 0.25, 0.05, key="t2_beta")
        t2_angle = st.number_input("200 DMA Angle >", value=5, key="t2_angle") # Independent angle control

    with st.sidebar.expander("Tab 3: Narrow Band"):
        t3_band_pct = st.slider("Price Band %", 0.01, 0.20, 0.05, 0.01, key="t3_band")
        t3_days = st.slider("Lookback Days", 5, 30, 10, key="t3_days")
        t3_angle = st.number_input("200 DMA Angle >", value=5, key="t3_angle")

    with st.sidebar.expander("Tab 4: Divergence Candidates (Slope filter)"):
        t4_angle = st.number_input("200 DMA Slope >", value=5, key="t4_angle")

    with st.sidebar.expander("Tab 5: 3DMA Angle"):
        t5_top_n = st.number_input("Top N Stocks", value=10, min_value=5, max_value=50, key="t5_top_n")
        t5_angle = st.number_input("200 DMA Slope >", value=5, key="t5_angle")
        
    with st.sidebar.expander("Tab 6 & 7: Crossovers"):
        crossover_angle = st.number_input("200 DMA Slope >", value=5, key="crossover_angle")
        proximity_pct = st.slider("Potential Proximity %", 0.1, 5.0, 2.0, 0.1, key="proximity_pct")
        
    with st.sidebar.expander("Tab 8: Volume Shockers"):
        vol_shock_threshold = st.slider("Volume Ratio (> x times avg)", 1.0, 10.0, 2.0, 0.1, key="vol_shock_threshold")
        min_volume = st.number_input("Min Average Volume", value=10000, step=10000, key="min_volume")
        t8_angle = st.number_input("200 DMA Slope >", value=5, key="t8_angle")

    # --- Filter Logic ---
    
    # helper to filter latest_df
    def get_display_data(tickers, include_vol_metrics=False):
        # Select relevant columns for display
        cols = ['Ticker', 'Close', '20DMA', '200DMA', '200DMA_LINE_ANGLE', 'RSI_14', 'RSI_14_angle', '3DMA_LINE_ANGLE', 'Beta']
        if include_vol_metrics:
            cols.extend(['Volume', 'VolumeRatio'])
            
        # Filter rows
        subset = latest_df[latest_df['Ticker'].isin(tickers)][cols].copy()
        
        # Rename for display
        rename_map = {
            'Ticker': 'Ticker', 'Close': 'Price', '20DMA': '20 DMA', '200DMA': '200 DMA', 
            '200DMA_LINE_ANGLE': '200DMA Angle', 'RSI_14': 'RSI', 'RSI_14_angle': 'RSI Angle', 
            '3DMA_LINE_ANGLE': '3DMA Angle', 'Beta': 'Beta'
        }
        if include_vol_metrics:
            rename_map.update({'Volume': 'Volume', 'VolumeRatio': 'Vol Ratio'})
            
        subset = subset.rename(columns=rename_map)
        # Round appropriate columns
        subset = subset.round(2)
        return subset

    # 1. Low RSI (<30) + 200DMA Slope > 5
    low_rsi_pos_slope_tickers = latest_df[
        (latest_df['RSI_14'] < t1_rsi) & 
        (latest_df['200DMA_LINE_ANGLE'] > t1_angle)
    ]['Ticker'].tolist()
    df_1 = get_display_data(low_rsi_pos_slope_tickers).sort_values(by='RSI', ascending=True)

    # 2. Low Beta + 200DMA Slope > 5
    beta_threshold = latest_df['Beta'].quantile(t2_beta_pct)
    low_beta_pos_slope_tickers = latest_df[
        (latest_df['Beta'] <= beta_threshold) & 
        (latest_df['200DMA_LINE_ANGLE'] > t2_angle)
    ]['Ticker'].tolist()
    df_2 = get_display_data(low_beta_pos_slope_tickers).sort_values(by='Beta', ascending=True)
    
    # 3. Narrow Price Band (2 weeks) + 200DMA Slope > 5
    narrow_band_tickers = []
    candidates = latest_df[latest_df['200DMA_LINE_ANGLE'] > t3_angle]['Ticker'].tolist()
    
    for ticker in candidates:
        stock_data = df[df['Ticker'] == ticker].sort_values('Date').tail(t3_days)
        if len(stock_data) < t3_days:
            continue
        min_price = stock_data['Close'].min()
        max_price = stock_data['Close'].max()
        if (max_price - min_price) / min_price < t3_band_pct:
            narrow_band_tickers.append(ticker)
    df_3 = get_display_data(narrow_band_tickers)

    # 4. Divergence Candidates with Slope Filter (200DMA Slope > 5)
    # Filter by Slope -> Sort by 3DMA Angle -> Top 10
    slope_filtered = latest_df[latest_df['200DMA_LINE_ANGLE'] > t4_angle]
    divergence_slope_tickers = slope_filtered.sort_values(by='3DMA_LINE_ANGLE', ascending=False).head(10)['Ticker'].tolist()
    df_4 = get_display_data(divergence_slope_tickers, include_vol_metrics=True).sort_values(by='3DMA Angle', ascending=False)
    
    # 5. Top 10 by 3DMA Angle (Unfiltered by Slope)
    divergence_tickers = latest_df.sort_values(by='3DMA_LINE_ANGLE', ascending=False).head(t5_top_n)['Ticker'].tolist()
    df_5 = get_display_data(divergence_tickers, include_vol_metrics=True).sort_values(by='3DMA Angle', ascending=False)

    # 6. Actual Bullish Crossovers (20DMA crosses above 200DMA)
    # latest > 200 AND prev <= 200
    actual_crossover_tickers = latest_df[
        (latest_df['20DMA'] > latest_df['200DMA']) &
        (latest_df['Prev_20DMA'] <= latest_df['Prev_200DMA']) &
        (latest_df['200DMA_LINE_ANGLE'] > crossover_angle)
    ]['Ticker'].tolist()
    df_6 = get_display_data(actual_crossover_tickers)

    # 7. Potential Bullish Crossovers
    # 20DMA < 200DMA (Below)
    # Distance < Threshold %
    # 20DMA Rising (Current > Prev)
    # 200DMA Positive Slope
    
    # Avoid division by zero
    mask_potential = (
        (latest_df['20DMA'] < latest_df['200DMA']) &
        (latest_df['200DMA'] != 0) &
        ((abs(latest_df['20DMA'] - latest_df['200DMA']) / latest_df['200DMA']) * 100 < proximity_pct) &
        (latest_df['20DMA'] > latest_df['Prev_20DMA']) &
        (latest_df['200DMA_LINE_ANGLE'] > crossover_angle)
    )
    potential_crossover_tickers = latest_df[mask_potential]['Ticker'].tolist()
    df_7 = get_display_data(potential_crossover_tickers)

    # 8. Volume Shockers
    # VolumeRatio > threshold
    # 20DayAvgVolume > min_volume
    # 200DMA_LINE_ANGLE > t8_angle
    
    shockers_mask = (
        (latest_df['VolumeRatio'] > vol_shock_threshold) &
        (latest_df['20DayAvgVolume'] > min_volume) &
        (latest_df['200DMA_LINE_ANGLE'] > t8_angle)
    )
    volume_shockers_tickers = latest_df[shockers_mask]['Ticker'].tolist()
    df_8 = get_display_data(volume_shockers_tickers, include_vol_metrics=True).sort_values(by='Vol Ratio', ascending=False)


    # --- Display ---
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Low RSI", 
        "Low Beta", 
        "Narrow Band", 
        "Divergence Slope",
        "High 3DMA Angle",
        "Actual Crossover",
        "Potential Crossover",
        "Volume Shockers"
    ])

    def render_tab_content(data_df, description, key_prefix):
        st.markdown(f"**{description}**")
        st.markdown(f"Found {len(data_df)} stocks.")
        
        if data_df.empty:
            st.info("No stocks matched this criteria.")
            return

        if key_prefix == "tab8" and 'Vol Ratio' in data_df.columns:
             display_data = data_df.style.background_gradient(subset=['Vol Ratio'], cmap='YlOrRd')
        elif key_prefix == "tab2" and 'Beta' in data_df.columns:
             display_data = data_df.style.background_gradient(subset=['Beta'], cmap='Blues')
        elif key_prefix == "tab1" and 'RSI' in data_df.columns:
             display_data = data_df.style.background_gradient(subset=['RSI'], cmap='RdYlGn_r')
        elif (key_prefix == "tab4" or key_prefix == "tab5") and '3DMA Angle' in data_df.columns:
             display_data = data_df.style.background_gradient(subset=['3DMA Angle'], cmap='Purples')
        else:
             display_data = data_df.style

        display_data = display_data.format(precision=2)

        # Sortable Data Grid with Selection
        # on_select="rerun" makes the app rerun when a row is selected
        # user can click headers to sort
        event = st.dataframe(
            display_data,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            key=f"grid_{key_prefix}"
        )

        # Check selection
        selected_rows = event.selection.rows
        
        if selected_rows:
            # Get the Ticker from the selected row index
            # data_df is what was displayed (filtered). Use iloc on it.
            selected_index = selected_rows[0]
            selected_ticker = data_df.iloc[selected_index]['Ticker']
            
            st.markdown("---")
            st.subheader(f"Analysis for {selected_ticker}")
            
            # Show metrics (using the row from the display df itself for speed)
            row = data_df.iloc[selected_index]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Close Price", f"{row['Price']:.2f}")
            col2.metric("RSI (14)", f"{row['RSI']:.2f}")
            col3.metric("200 DMA Angle", f"{row['200DMA Angle']:.2f}°")
            col4.metric("3 DMA Angle", f"{row['3DMA Angle']:.2f}°")
            col5.metric("Beta", f"{row['Beta']:.2f}")
            
            plot_charts(df, selected_ticker, timeframe_days, candle_type, show_bollinger, show_volume_profile)
        else:
            st.info("👆 Click a row in the table above to view the chart.")

    with tab1:
        render_tab_content(df_1, f"RSI < {t1_rsi} and 200 DMA Slope > {t1_angle}°", "tab1")

    with tab2:
        render_tab_content(df_2, f"Low Beta (Bottom {int(t2_beta_pct*100)}%) and 200 DMA Slope > {t2_angle}°", "tab2")

    with tab3:
        render_tab_content(df_3, f"Narrow Price Band ({int(t3_band_pct*100)}% range/{t3_days} days) and 200 DMA Slope > {t3_angle}°", "tab3")
        
    with tab4:
        render_tab_content(df_4, f"Top 10 High 3DMA Angle (Divergence) with 200 DMA Slope > {t4_angle}°", "tab4")
        
    with tab5:
        render_tab_content(df_5, f"Top {t5_top_n} Stocks by Highest 3DMA Angle (200DMA Slope > {t5_angle}°)", "tab5")

    with tab6:
        render_tab_content(df_6, f"Actual Bullish Crossover (20DMA crosses 200DMA) + Slope > {crossover_angle}°", "tab6")

    with tab7:
        render_tab_content(df_7, f"Potential Bullish Crossover (Gap < {proximity_pct}%) + Slope > {crossover_angle}°", "tab7")

    with tab8:
        render_tab_content(df_8, f"Volume > {vol_shock_threshold}x Avg AND Avg Vol > {min_volume} AND Slope > {t8_angle}°", "tab8")

if __name__ == "__main__":
    main()
