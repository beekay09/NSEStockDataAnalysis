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

# Page configuration
st.set_page_config(
    page_title="NSE Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
DATA_FILE = "nse_stock_data_with_metrics_v2.csv"
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
        # Calculate Previous Day's DMA Slopes for Bottoming Logic (3, 20, 200)
        for dma in [3, 20, 200]:
            if f'{dma}DMA_SLOPE' in df.columns:
                df[f'Prev_{dma}DMA_SLOPE'] = df.groupby('Ticker')[f'{dma}DMA_SLOPE'].shift(1)

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
        "DMA Bottoming"
    ]
    
    # Custom CSS to make radio buttons look like tabs/pills
    st.markdown("""
    <style>
        div.row-widget.stRadio > div {
            flex-direction: row;
            justify-content: center;
            background-color: #262730;
            padding: 10px;
            border-radius: 10px;
            overflow-x: auto;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: #0e1117;
            padding: 5px 15px;
            border-radius: 15px;
            margin: 0 5px;
            border: 1px solid #464b5d;
        }
    </style>
    """, unsafe_allow_html=True)
    
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
            
            plot_charts(df, selected_ticker, timeframe_days, candle_type, show_bollinger, show_volume_profile)
        else:
            st.info("👆 Select a row (checkmark) to view the chart.")

    # Render Active Tab Content Only
    if selected_tab == "Low RSI":
        low_rsi_pos_slope_tickers = latest_df[
            (latest_df['RSI_14'] < t1_rsi) & 
            (latest_df['200DMA_SLOPE'] > t1_angle)
        ]['Ticker'].tolist()
        df_1 = get_display_data(low_rsi_pos_slope_tickers).sort_values(by=['RSI', 'RSI Slope'], ascending=[True, True])
        
        doc_1 = """
        **Low RSI Strategy**
        
        Identifies stocks that are potentially **oversold** (Low RSI) but are still in a **long-term uptrend** (Positive 200 DMA Slope).
        
        *   **Goal**: Catch pullbacks in strong stocks.
        *   **Key Filters**:
            *   `RSI Threshold`: Stocks with RSI below this value.
            *   `200 DMA Angle`: Ensures the long-term trend is positive.
        """
        render_tab_content(df_1, f"RSI < {t1_rsi} and 200 DMA Slope > {t1_angle}°", "tab1", doc_1)
        
    elif selected_tab == "Low Beta":
        beta_threshold = latest_df['Beta'].quantile(t2_beta_pct)
        low_beta_pos_slope_tickers = latest_df[
            (latest_df['Beta'] <= beta_threshold) & 
            (latest_df['200DMA_SLOPE'] > t2_angle)
        ]['Ticker'].tolist()
        df_2 = get_display_data(low_beta_pos_slope_tickers).sort_values(by='Beta', ascending=True)
        
        doc_2 = """
        **Low Beta Strategy**
        
        Finds **low-volatility stocks** (Low Beta) that are in a **steady uptrend**.
        
        *   **Goal**: Identify conservative entry points in stable stocks.
        *   **Key Filters**:
            *   `Beta Percentile`: Selects the bottom percentage of stocks by Beta.
            *   `200 DMA Angle`: Ensures the long-term trend is positive.
        """
        render_tab_content(df_2, f"Low Beta (Bottom {int(t2_beta_pct*100)}%) and 200 DMA Slope > {t2_angle}°", "tab2", doc_2)

    elif selected_tab == "Narrow Band":
        narrow_band_tickers = []
        candidates = latest_df[latest_df['200DMA_SLOPE'] > t3_angle]['Ticker'].tolist()
        for ticker in candidates:
            stock_data = df[df['Ticker'] == ticker].sort_values('Date').tail(t3_days)
            if len(stock_data) < t3_days:
                continue
            min_price = stock_data['Close'].min()
            max_price = stock_data['Close'].max()
            if (max_price - min_price) / min_price < t3_band_pct:
                narrow_band_tickers.append(ticker)
        df_3 = get_display_data(narrow_band_tickers)
        
        doc_3 = """
        **Narrow Price Band Strategy**
        
        Detects stocks that are **consolidating** within a tight price range (volatility contraction) while maintaining a long-term uptrend.
        
        *   **Goal**: Anticipate a potential breakout from consolidation.
        *   **Key Filters**:
            *   `Price Band %`: Maximum allowed percentage difference between High and Low over the lookback period.
            *   `Lookback Days`: Number of days the price has stayed in this range.
        """
        render_tab_content(df_3, f"Narrow Price Band ({int(t3_band_pct*100)}% range/{t3_days} days) and 200 DMA Slope > {t3_angle}°", "tab3", doc_3)

    elif selected_tab == "Divergence Slope":
        slope_filtered = latest_df[latest_df['200DMA_SLOPE'] > t4_angle]
        divergence_slope_tickers = slope_filtered.sort_values(by='3DMA_SLOPE', ascending=False).head(10)['Ticker'].tolist()
        df_4 = get_display_data(divergence_slope_tickers, include_vol_metrics=True).sort_values(by='3DMA Slope', ascending=False)
        
        doc_4 = """
        **Divergence Slope Strategy**
        
        Highlights stocks where the **short-term trend (3DMA Slope)** is very strong compared to peers, indicating strong immediate momentum.
        
        *   **Goal**: Identify stocks with explosive short-term momentum supported by a long-term trend.
        *   **Key Filters**:
            *   `200 DMA Slope`: Minimum angle for the long-term trend.
            *   Ranks by `3DMA Slope` descending (Top 10).
        """
        render_tab_content(df_4, f"Top 10 High 3DMA Slope (Divergence) with 200 DMA Slope > {t4_angle}°", "tab4", doc_4)

    elif selected_tab == "High DMA Angle":
        target_col = f"{selected_dma}DMA_SLOPE"
        display_name = f"{selected_dma}DMA Slope"
        
        divergence_tickers = latest_df.sort_values(by=target_col, ascending=False).head(t5_top_n)['Ticker'].tolist()
        df_5 = get_display_data(divergence_tickers, include_vol_metrics=True).sort_values(by=display_name, ascending=False)
        
        doc_5 = f"""
        **High DMA Angle Ranking**
        
        Ranks stocks purely by the **steepness of their trend** for the selected Annual/DMA timeframe.
        
        *   **Goal**: Find the strongest trending stocks right now.
        *   **Key Filters**:
            *   `Select DMA`: Choose between Short (3), Medium (20), or Long (200) term trends.
            *   `Top N`: Number of top stocks to display.
        """
        render_tab_content(df_5, f"Top {t5_top_n} Stocks by Highest {display_name} (200DMA Slope > {t5_angle}°)", "tab5", doc_5)

    elif selected_tab == "Slope Difference":
        latest_df['SlopeDiff'] = latest_df['20DMA_SLOPE'] - latest_df['200DMA_SLOPE']
        
        mask = (latest_df['SlopeDiff'] >= t6_min_diff) & (latest_df['200DMA_SLOPE'] > t6_min_200_slope)
            
        slope_diff_tickers = latest_df[mask].sort_values(by='SlopeDiff', ascending=False).head(t6_top_n)['Ticker'].tolist()
        df_diff = get_display_data(slope_diff_tickers)
        # We need to add the SlopeDiff column to the display dataframe for clarity, 
        # but get_display_data filters columns. 
        # Let's just rely on the user seeing 20DMA Slope and 200DMA Slope and doing the math, 
        # OR we can add it to the dataframe afterwards.
        # Let's add it afterwards.
        # Recalculate for the subset to be safe/easy
        df_diff['Slope Diff'] = df_diff['20DMA Slope'] - df_diff['200DMA Slope']
        
        doc_6 = """
        **Slope Difference Strategy**
        
        Focuses on the **spread** between the Short-Term (20DMA) Slope and the Long-Term (200DMA) Slope.
        
        *   **Goal**: Identify meaningful acceleration where the short-term trend is significantly outpacing the long-term trend.
        *   **Key Filters**:
            *   `Min Slope Difference`: Minimum spread required (20DMA Slope - 200DMA Slope).
            *   `200 DMA Slope`: Ensures the base trend is positive.
        """
        render_tab_content(df_diff, f"Top {t6_top_n} Stocks by (20DMA Slope - 200DMA Slope) >= {t6_min_diff}", "tab_slope_diff", doc_6)

    elif selected_tab == "Actual Crossover":
        actual_crossover_tickers = latest_df[
            (latest_df['20DMA'] > latest_df['200DMA']) &
            (latest_df['Prev_20DMA'] <= latest_df['Prev_200DMA']) &
            (latest_df['200DMA_SLOPE'] > crossover_angle)
        ]['Ticker'].tolist()
        df_6 = get_display_data(actual_crossover_tickers)
        
        doc_7 = """
        **Actual Crossover (Golden Cross)**
        
        Identifies stocks where the **20 DMA crossed ABOVE the 200 DMA** today.
        
        *   **Goal**: Catch the start of a potential major uptrend signal.
        *   **Key Filters**:
            *   `200 DMA Slope`: Ensures the long-term trend is not downward.
            *   Logic: Today 20DMA > 200 DMA AND Yesterday 20DMA <= 200DMA.
        """
        render_tab_content(df_6, f"Actual Bullish Crossover (20DMA crosses 200DMA) + Slope > {crossover_angle}°", "tab6", doc_7)

    elif selected_tab == "Potential Crossover":
        mask_potential = (
            (latest_df['20DMA'] < latest_df['200DMA']) &
            (latest_df['200DMA'] != 0) &
            ((abs(latest_df['20DMA'] - latest_df['200DMA']) / latest_df['200DMA']) * 100 < proximity_pct) &
            (latest_df['20DMA'] > latest_df['Prev_20DMA']) &
            (latest_df['200DMA_SLOPE'] > crossover_angle)
        )
        potential_crossover_tickers = latest_df[mask_potential]['Ticker'].tolist()
        df_7 = get_display_data(potential_crossover_tickers)
        
        doc_8 = """
        **Potential Crossover / Kissing Distance**
        
        Finds stocks where the 20 DMA is **approaching** the 200 DMA from below and is very close.
        
        *   **Goal**: Anticipate a crossover before it happens.
        *   **Key Filters**:
            *   `Proximity %`: How close the 20DMA is to the 200DMA (in %).
            *   `200 DMA Slope`: Ensures positive long-term background.
        """
        render_tab_content(df_7, f"Potential Bullish Crossover (Gap < {proximity_pct}%) + Slope > {crossover_angle}°", "tab7", doc_8)

    elif selected_tab == "Volume Shockers":
        shockers_mask = (
            (latest_df['VolumeRatio'] > vol_shock_threshold) &
            (latest_df['20DayAvgVolume'] > min_volume) &
            (latest_df['200DMA_SLOPE'] > t8_angle)
        )
        volume_shockers_tickers = latest_df[shockers_mask]['Ticker'].tolist()
        df_8 = get_display_data(volume_shockers_tickers, include_vol_metrics=True).sort_values(by='Vol Ratio', ascending=False)
        
        doc_9 = """
        **Volume Shockers**
        
        Identifies stocks with **unusual volume spikes** compared to their **20-day average**.
        
        *   **Goal**: Detect institutional interest, breakouts, or news-driven moves.
        *   **Key Filters**:
            *   `Volume Ratio`: Current Volume / 20-Day Average Volume.
            *   `Min Avg Volume`: Filter out illiquid stocks.
        """
        render_tab_content(df_8, f"Volume > {vol_shock_threshold}x Avg AND Avg Vol > {min_volume} AND Slope > {t8_angle}°", "tab8", doc_9)

    elif selected_tab == "DMA Bottoming":
        cur_col = f"{selected_dma_bot}DMA_SLOPE"
        prev_col = f"Prev_{selected_dma_bot}DMA_SLOPE"
        display_name_bot = f"{selected_dma_bot}DMA Slope"
        
        bottoming_mask = (
            (latest_df[prev_col] <= t9_prev_max) &
            (latest_df[cur_col] > latest_df[prev_col]) &
            (latest_df[cur_col] >= t9_min_angle) &
            (latest_df[cur_col] <= t9_max_angle)
        )
        bottoming_tickers = latest_df[bottoming_mask]['Ticker'].tolist()
        df_9 = get_display_data(bottoming_tickers, include_vol_metrics=True).sort_values(by=display_name_bot, ascending=True)
        
        doc_10 = f"""
        **DMA Bottoming / Turning Up**
        
        Identifies stocks where the slope of the **{selected_dma_bot} DMA** is **curving upwards**.
        
        *   **Goal**: Spot trend reversals (end of downtrend or correction) early.
        *   **Key Filters**:
            *   Previously negative or flat slope -> Now positive or less negative slope.
            *   Min/Max Angle constraints.
        """
        render_tab_content(df_9, f"{selected_dma_bot}DMA Bottoming: Prev Angle <= {t9_prev_max}° → Current [{t9_min_angle}°, {t9_max_angle}°] (Turning Up)", "tab9", doc_10)

if __name__ == "__main__":
    main()
