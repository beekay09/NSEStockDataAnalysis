"""
run_filters.py - Batch runner for all stock analysis filters.
Reads filter_config.json, runs each enabled filter, and writes results to an Excel file
with one sheet per filter.

Usage:
    python run_filters.py
    python run_filters.py --config my_config.json
    python run_filters.py --output my_results.xlsx
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
import sys
import time
import datetime

import yfinance as yf
import analysis_logic as al
import metrics as mt
import add_metrics


# --- Fundamental data fields to fetch from yfinance ---
FUNDAMENTAL_FIELDS = {
    'sector':               'Sector',
    'industry':             'Industry',
    'marketCap':            'Mkt Cap',
    'trailingPE':           'P/E',
    'priceToBook':          'P/B',
    'trailingEps':          'EPS',
    'dividendYield':        'Div Yield %',
    'returnOnEquity':       'ROE %',
    'debtToEquity':         'D/E',
    'fiftyTwoWeekHigh':     '52W High',
    'fiftyTwoWeekLow':      '52W Low',
    'revenueGrowth':        'Rev Growth %',
    'earningsGrowth':       'Earnings Growth %',
    'bookValue':            'Book Value',
    'operatingMargins':     'Op Margin %',
    'freeCashflow':         'Free CF',
}


def enrich_with_fundamentals(flat_df):
    """Fetch yfinance fundamentals for unique tickers and merge into flat_df."""
    unique_tickers = flat_df['Ticker'].unique().tolist()
    total = len(unique_tickers)
    print(f"Fetching fundamentals for {total} unique tickers...")

    fund_rows = []
    failed = []
    for i, ticker in enumerate(unique_tickers, 1):
        # Append .NS for NSE tickers if not already present
        yf_ticker = ticker if '.' in ticker else f"{ticker}.NS"

        row = {'Ticker': ticker}
        success = False

        for attempt in range(3):  # up to 3 retries
            try:
                info = yf.Ticker(yf_ticker).info
                if info and len(info) > 5:  # valid response (not just error stub)
                    for yf_key, col_name in FUNDAMENTAL_FIELDS.items():
                        val = info.get(yf_key)
                        if val is not None and yf_key in (
                            'dividendYield', 'returnOnEquity', 'revenueGrowth',
                            'earningsGrowth', 'operatingMargins'
                        ):
                            val = round(val * 100, 2)
                        if val is not None and yf_key == 'marketCap':
                            val = round(val / 1e7, 0)  # INR to Cr
                        row[col_name] = val
                    success = True
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5)  # brief pause before retry
                continue

        if not success:
            failed.append(ticker)

        fund_rows.append(row)

        # Progress indicator every 25 tickers
        if i % 25 == 0 or i == total:
            print(f"  [{i:3d}/{total}] fetched")

        # Small delay to avoid rate limiting
        if i % 5 == 0:
            time.sleep(0.1)

    if failed:
        print(f"  Warning: {len(failed)} tickers failed: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    fund_df = pd.DataFrame(fund_rows)
    flat_df = flat_df.merge(fund_df, on='Ticker', how='left')
    return flat_df


# --- Data Loading (mirrors App.py logic, without Streamlit) ---

DATA_FILE = "nse_stock_data_with_metrics_v2.csv"
DEFAULT_CONFIG = "filter_config.json"
DEFAULT_OUTPUT = "filter_results.xlsx"
FRESHNESS_HOURS = 20


def ensure_fresh_data():
    """Check if metrics CSV is fresh. If stale or missing, re-run the pipeline."""
    if os.path.exists(DATA_FILE):
        file_mod_time = os.path.getmtime(DATA_FILE)
        hours_old = (time.time() - file_mod_time) / 3600
        if hours_old <= FRESHNESS_HOURS:
            print(f"Data is fresh ({hours_old:.1f} hrs old, limit: {FRESHNESS_HOURS}h). Skipping refresh.")
            return
        else:
            print(f"Data is STALE ({hours_old:.1f} hrs old, limit: {FRESHNESS_HOURS}h).")
    else:
        print(f"Data file '{DATA_FILE}' not found.")

    print("Running data fetch + metrics pipeline...")
    print("=" * 60)
    add_metrics.main()
    print("=" * 60)
    print("Pipeline complete. Proceeding with filters.\n")

def load_data():
    """Load and preprocess the stock data (same as App.py but without st.cache)."""
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])

    # Daily returns for Beta
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()

    # Previous day's DMA values for crossover logic
    df['Prev_20DMA'] = df.groupby('Ticker')['20DMA'].shift(1)
    df['Prev_200DMA'] = df.groupby('Ticker')['200DMA'].shift(1)

    # Previous day's DMA slopes for bottoming logic
    for dma in [3, 20, 200]:
        if f'{dma}DMA_SLOPE' in df.columns:
            df[f'Prev_{dma}DMA_SLOPE'] = df.groupby('Ticker')[f'{dma}DMA_SLOPE'].shift(1)

    # 20-day volume SMA
    df['Volume_20SMA'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20).mean()
    )

    # 20-day average volume
    df['20DayAvgVolume'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(20).mean()
    )

    # Benchmark
    benchmark_df = df[df['Ticker'] == 'SBINEQWETF.NS'].copy()
    if benchmark_df.empty and 'SBINEQWETF' in df['Ticker'].unique():
        benchmark_df = df[df['Ticker'] == 'SBINEQWETF'].copy()

    return df, benchmark_df


def prepare_latest(df):
    """Extract latest row per ticker + compute beta and volume ratio."""
    print("Computing beta metrics...")
    beta_map = mt.calculate_beta_metrics(df)

    latest_df = df.sort_values(by=['Ticker', 'Date']).groupby('Ticker').tail(1).copy()
    latest_df['Beta'] = latest_df['Ticker'].map(beta_map)
    latest_df['VolumeRatio'] = latest_df['Volume'] / latest_df['20DayAvgVolume']

    return latest_df


def build_display_df(latest_df, tickers, extra_cols=None):
    """Build a standardized display DataFrame for a list of tickers."""
    cols = [
        'Ticker', 'Close', 'RSI_14', 'RSI_14_SLOPE',
        'Beta', 'Volume', 'VolumeRatio',
        '3DMA_SLOPE', '20DMA_SLOPE', '200DMA_SLOPE',
        '20DMA', '200DMA'
    ]
    # Only keep columns that exist
    cols = [c for c in cols if c in latest_df.columns]

    subset = latest_df[latest_df['Ticker'].isin(tickers)][cols].copy()

    rename_map = {
        'Close': 'Price',
        'RSI_14': 'RSI', 'RSI_14_SLOPE': 'RSI Slope',
        'VolumeRatio': 'Vol Ratio',
        '3DMA_SLOPE': '3DMA Slope',
        '20DMA_SLOPE': '20DMA Slope',
        '200DMA_SLOPE': '200DMA Slope',
        '20DMA': '20 DMA', '200DMA': '200 DMA'
    }
    subset = subset.rename(columns=rename_map)

    # Add Price/200DMA %
    if 'Price' in subset.columns and '200 DMA' in subset.columns:
        subset['Price/200DMA %'] = (
            (subset['Price'] - subset['200 DMA']) / subset['200 DMA'] * 100
        ).round(2)

    subset = subset.round(2)
    return subset


# --- Filter runners ---

def run_low_rsi(cfg, latest_df, df, **kwargs):
    c = cfg["low_rsi"]
    tickers, _ = al.get_low_rsi_tickers(
        latest_df, c["rsi_threshold"], c["min_200dma_slope"],
        c.get("rsi_turning_up", False), c.get("volume_spike", False)
    )
    result = build_display_df(latest_df, tickers)
    return result.sort_values(by=['RSI'], ascending=True) if not result.empty else result


def run_low_beta(cfg, latest_df, df, **kwargs):
    c = cfg["low_beta"]
    tickers, _ = al.get_low_beta_tickers(
        latest_df, c["beta_percentile"], c["min_200dma_slope"],
        c.get("near_200dma", False), c.get("rsi_turning_up", False)
    )
    result = build_display_df(latest_df, tickers)
    return result.sort_values(by='Beta', ascending=True) if not result.empty else result


def run_narrow_band(cfg, latest_df, df, **kwargs):
    c = cfg["narrow_band"]
    tickers, _ = al.get_narrow_band_tickers(
        latest_df, df, c["band_pct"], c["lookback_days"], c["min_200dma_slope"]
    )
    return build_display_df(latest_df, tickers)


def run_divergence_slope(cfg, latest_df, df, **kwargs):
    c = cfg["divergence_slope"]
    tickers, _ = al.get_divergence_slope_tickers(
        latest_df, c["min_200dma_slope"],
        c.get("rsi_check", False), c.get("volume_check", False),
        c.get("max_rsi", 70), c.get("min_vol_ratio", 1.0)
    )
    result = build_display_df(latest_df, tickers)
    return result.sort_values(by='3DMA Slope', ascending=False) if not result.empty else result


def run_high_dma_angle(cfg, latest_df, df, **kwargs):
    config_key = kwargs.get("config_key", "high_dma_angle_3")
    c = cfg[config_key]
    tickers, _ = al.get_high_dma_angle_tickers(
        latest_df, c["dma"], c["top_n"], c["min_200dma_slope"]
    )
    sort_col = f"{c['dma']}DMA Slope"
    result = build_display_df(latest_df, tickers)
    if not result.empty and sort_col in result.columns:
        result = result.sort_values(by=sort_col, ascending=False)
    return result


def run_slope_difference(cfg, latest_df, df, **kwargs):
    c = cfg["slope_difference"]
    tickers, _ = al.get_slope_difference_tickers(
        latest_df, c["min_diff"], c["min_200dma_slope"], c["top_n"]
    )
    result = build_display_df(latest_df, tickers)
    if not result.empty:
        slope_diff_map = latest_df.set_index('Ticker')['SlopeDiff']
        result['Slope Diff'] = result['Ticker'].map(slope_diff_map).round(2)
    return result


def run_price_crossover(cfg, latest_df, df, **kwargs):
    config_key = kwargs.get("config_key", "price_crossover_200")
    c = cfg[config_key]
    ct = c.get("crossover_type", "Price x 200DMA")
    lookback = c.get("lookback_days", 1)
    min_slope = c.get("min_slope", 5)

    dma_map = {
        "Price x 200DMA": 200,
        "Price x 100DMA": 100,
        "Price x 20DMA": 20,
    }
    dma_period = dma_map.get(ct, 200)

    tickers, crossover_info, _ = al.get_price_dma_crossover_tickers(
        latest_df, df, min_slope, lookback, dma_period=dma_period
    )
    result = build_display_df(latest_df, tickers)
    if not result.empty and lookback > 1 and crossover_info:
        result['Crossover Date'] = result['Ticker'].map(crossover_info)
        result['Crossover Date'] = pd.to_datetime(result['Crossover Date']).dt.strftime('%Y-%m-%d')
    return result


def run_potential_crossover(cfg, latest_df, df, **kwargs):
    c = cfg["potential_crossover"]
    tickers, _ = al.get_potential_crossover_tickers(
        latest_df, c["proximity_pct"], c["min_200dma_slope"]
    )
    return build_display_df(latest_df, tickers)


def run_volume_shockers(cfg, latest_df, df, **kwargs):
    c = cfg["volume_shockers"]
    tickers, _ = al.get_volume_shockers_tickers(
        latest_df, c["vol_ratio_threshold"], c["min_avg_volume"],
        c["min_200dma_slope"], c.get("sentiment", "Both")
    )
    result = build_display_df(latest_df, tickers)
    return result.sort_values(by='Vol Ratio', ascending=False) if not result.empty else result


def run_dma_bottoming(cfg, latest_df, df, **kwargs):
    c = cfg["dma_bottoming"]
    tickers, _ = al.get_dma_bottoming_tickers(
        latest_df, c["dma"], c["min_angle"], c["max_angle"], c["prev_max_angle"]
    )
    sort_col = f"{c['dma']}DMA Slope"
    result = build_display_df(latest_df, tickers)
    if not result.empty and sort_col in result.columns:
        result = result.sort_values(by=sort_col, ascending=True)
    return result


def run_macd_crossover(cfg, latest_df, df, **kwargs):
    c = cfg["macd_crossover"]
    tickers, _ = al.get_macd_crossover_tickers(
        latest_df, df, c["min_200dma_slope"], c.get("lookback_days", 3)
    )
    return build_display_df(latest_df, tickers)


def run_slope_crossover(cfg, latest_df, df, **kwargs):
    c = cfg["slope_crossover"]
    tickers, _ = al.get_slope_crossover_tickers(
        latest_df, df, c.get("lookback_days", 3), c.get("min_20dma_slope", 0)
    )
    result = build_display_df(latest_df, tickers)
    return result.sort_values(by='20DMA Slope', ascending=False) if not result.empty else result


def run_relative_strength(cfg, latest_df, df, benchmark_df=None, **kwargs):
    c = cfg["relative_strength"]
    tickers, _ = al.get_rs_strong_tickers(
        latest_df, df, benchmark_df,
        c.get("min_rs_slope", 0), c.get("min_200dma_slope", 0)
    )
    result = build_display_df(latest_df, tickers)
    top_n = c.get("top_n", 0)
    if top_n and not result.empty:
        result = result.sort_values(by='200DMA Slope', ascending=False).head(top_n)
    return result


def run_strong_adx(cfg, latest_df, df, **kwargs):
    c = cfg["strong_adx"]
    tickers, _ = al.get_strong_adx_tickers(
        latest_df, df, c.get("adx_threshold", 25),
        c.get("buy_side_only", True), c.get("crossover_only", False)
    )
    result = build_display_df(latest_df, tickers)
    top_n = c.get("top_n", 0)
    if top_n and not result.empty:
        result = result.sort_values(by='200DMA Slope', ascending=False).head(top_n)
    return result


def run_heikin_ashi_turn(cfg, latest_df, df, **kwargs):
    c = cfg["heikin_ashi_turn"]
    tickers, _ = al.get_heikin_ashi_turnover_tickers(
        latest_df, df, c.get("min_200dma_slope", 0)
    )
    result = build_display_df(latest_df, tickers)
    top_n = c.get("top_n", 200)
    if not result.empty:
        if '200DMA Slope' in result.columns:
            result = result.sort_values(by='200DMA Slope', ascending=False)
        if top_n:
            result = result.head(top_n)
    return result


# --- Main ---

# Registry: maps config key -> (runner function, sheet name, needs_benchmark)
FILTER_REGISTRY = {
    "low_rsi":              (run_low_rsi,              "Low RSI",              False),
    "low_beta":             (run_low_beta,             "Low Beta",             False),
    "narrow_band":          (run_narrow_band,          "Narrow Band",          False),
    "divergence_slope":     (run_divergence_slope,     "Divergence Slope",     False),
    "high_dma_angle_3":     (run_high_dma_angle,       "High DMA Angle (3)",   False),
    "high_dma_angle_20":    (run_high_dma_angle,       "High DMA Angle (20)",  False),
    "high_dma_angle_200":   (run_high_dma_angle,       "High DMA Angle (200)", False),
    "slope_difference":     (run_slope_difference,     "Slope Difference",     False),
    "price_crossover_200":  (run_price_crossover,      "Price x 200DMA",       False),
    "price_crossover_100":  (run_price_crossover,      "Price x 100DMA",       False),
    "price_crossover_20":   (run_price_crossover,      "Price x 20DMA",        False),
    "potential_crossover":  (run_potential_crossover,   "Potential Crossover",  False),
    "volume_shockers":      (run_volume_shockers,      "Volume Shockers",      False),
    "dma_bottoming":        (run_dma_bottoming,        "DMA Bottoming",        False),
    "macd_crossover":       (run_macd_crossover,       "MACD Crossover",       False),
    "slope_crossover":      (run_slope_crossover,      "Slope Crossover",      False),
    "relative_strength":    (run_relative_strength,    "Relative Strength",    True),
    "strong_adx":           (run_strong_adx,           "Strong ADX",           False),
    "heikin_ashi_turn":     (run_heikin_ashi_turn,     "Heikin Ashi Turn",     False),
}




SIGNAL_WEIGHTS = {
    "Low RSI":              3,   # Oversold bounce potential
    "Low Beta":             1,   # Defensive, less actionable
    "Narrow Band":          2,   # Consolidation breakout setup
    "Divergence Slope":     2,   # Short-term momentum
    "High DMA Angle (3)":   1,   # Very short-term, noisy
    "High DMA Angle (20)":  2,   # Medium-term trend strength
    "High DMA Angle (200)": 2,   # Long-term trend strength
    "Slope Difference":     2,   # Momentum acceleration
    "Price x 200DMA":       4,   # Major trend reversal signal
    "Price x 100DMA":       3,   # Medium-term trend change
    "Price x 20DMA":        2,   # Short-term trend change
    "Potential Crossover":  2,   # Approaching crossover
    "Volume Shockers":      3,   # Institutional interest
    "DMA Bottoming":        3,   # Trend reversal early signal
    "MACD Crossover":       3,   # Classic momentum signal
    "Slope Crossover":      2,   # Slope momentum shift
    "Relative Strength":    2,   # Outperforming market
    "Strong ADX":           2,   # Strong trend confirmation
    "Heikin Ashi Turn":     3,   # Trend reversal early signal
}


# --- Signal categories for grouping ---
SIGNAL_CATEGORIES = {
    "Trend Reversal":  ["Price x 200DMA", "Price x 100DMA", "DMA Bottoming", "Potential Crossover", "Heikin Ashi Turn"],
    "Momentum":        ["MACD Crossover", "Divergence Slope", "Slope Difference", "Slope Crossover"],
    "Oversold":        ["Low RSI", "Price x 20DMA"],
    "Trend Strength":  ["High DMA Angle (20)", "High DMA Angle (200)", "Strong ADX", "Relative Strength"],
    "Volume":          ["Volume Shockers"],
    "Consolidation":   ["Narrow Band"],
}


def run_all_filters(cfg):
    """Run all enabled filters and return flat DataFrame + results dict."""
    df, benchmark_df = load_data()
    latest_df = prepare_latest(df)
    data_date = df['Date'].max().strftime('%Y-%m-%d')

    results = {}
    for key, (runner, sheet_name, needs_benchmark) in FILTER_REGISTRY.items():
        filter_cfg = cfg.get(key, {})
        if not filter_cfg.get("enabled", True):
            continue
        try:
            if needs_benchmark:
                result_df = runner(cfg, latest_df, df, benchmark_df, config_key=key)
            else:
                result_df = runner(cfg, latest_df, df, config_key=key)
            if not result_df.empty:
                results[sheet_name] = result_df
        except Exception:
            pass

    return results, latest_df, data_date


def score_stocks(results):
    """Score each stock based on filter convergence and signal weights."""
    ticker_signals = {}  # {ticker: [(filter_name, weight), ...]}

    for filter_name, result_df in results.items():
        weight = SIGNAL_WEIGHTS.get(filter_name, 1)
        for ticker in result_df['Ticker'].tolist():
            if ticker not in ticker_signals:
                ticker_signals[ticker] = []
            ticker_signals[ticker].append((filter_name, weight))

    # Build scored DataFrame
    scored = []
    for ticker, signals in ticker_signals.items():
        total_weight = sum(w for _, w in signals)
        filter_count = len(signals)
        filter_names = [s[0] for s in signals]

        # Category coverage
        categories_hit = set()
        for cat_name, cat_filters in SIGNAL_CATEGORIES.items():
            if any(f in filter_names for f in cat_filters):
                categories_hit.add(cat_name)

        scored.append({
            'Ticker': ticker,
            'Score': total_weight,
            'Filters': filter_count,
            'Categories': len(categories_hit),
            'Category List': ', '.join(sorted(categories_hit)),
            'Signals': ' | '.join(filter_names),
        })

    scored_df = pd.DataFrame(scored)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(
            by=['Score', 'Categories', 'Filters'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return scored_df


def classify_opportunity(row):
    """Classify the type of opportunity based on signals present."""
    signals = row['Signals']
    categories = row['Category List']

    if 'Trend Reversal' in categories and 'Momentum' in categories:
        return 'STRONG REVERSAL'
    elif 'Trend Reversal' in categories and 'Volume' in categories:
        return 'BREAKOUT'
    elif 'Oversold' in categories and 'Trend Strength' in categories:
        return 'BOUNCE PLAY'
    elif 'Momentum' in categories and 'Trend Strength' in categories:
        return 'MOMENTUM RIDE'
    elif 'Consolidation' in categories and 'Momentum' in categories:
        return 'BREAKOUT SETUP'
    elif 'Trend Reversal' in categories:
        return 'EARLY REVERSAL'
    elif 'Momentum' in categories:
        return 'MOMENTUM'
    elif 'Oversold' in categories:
        return 'OVERSOLD'
    elif 'Trend Strength' in categories:
        return 'TRENDING'
    else:
        return 'WATCH'


def generate_report(scored_df, results, data_date, top_n=15):
    """Generate a text report of top opportunities."""
    lines = []
    lines.append("=" * 70)
    lines.append("  STOCK OPPORTUNITY ANALYSIS REPORT")
    lines.append(f"  Data Date: {data_date}")
    lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # Summary
    lines.append(f"\nFilters Run: {len(results)}")
    lines.append(f"Unique Stocks Found: {len(scored_df)}")

    filter_summary = [(name, len(df)) for name, df in results.items()]
    filter_summary.sort(key=lambda x: x[1], reverse=True)
    lines.append("\nFilter Results Summary:")
    for name, count in filter_summary:
        lines.append(f"  {name:30s}  {count:3d} stocks")

    # Top opportunities
    top = scored_df.head(top_n)
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  TOP {top_n} OPPORTUNITIES (Ranked by Convergence Score)")
    lines.append("=" * 70)

    for idx, row in top.iterrows():
        opp_type = classify_opportunity(row)
        lines.append(f"\n{'~' * 60}")
        lines.append(f"  #{idx+1}  {row['Ticker']}  [{row.get('Sector', 'N/A')}]")
        
        fund_str = []
        if pd.notna(row.get('Mkt Cap')): fund_str.append(f"MktCap: {row['Mkt Cap']}Cr")
        if pd.notna(row.get('P/E')): fund_str.append(f"PE: {row['P/E']}")
        if pd.notna(row.get('ROE %')): fund_str.append(f"ROE: {row['ROE %']}%")
        if pd.notna(row.get('D/E')): fund_str.append(f"D/E: {row['D/E']}")
        if fund_str:
            lines.append(f"  Fundamentals: {' | '.join(fund_str)}")
            
        tech_str = []
        if pd.notna(row.get('Price')): tech_str.append(f"Price: {row['Price']}")
        if pd.notna(row.get('RSI')): tech_str.append(f"RSI: {row['RSI']:.1f}")
        if pd.notna(row.get('Vol Ratio')): tech_str.append(f"Volx: {row['Vol Ratio']:.1f}")
        if pd.notna(row.get('Price/200DMA %')): tech_str.append(f"vs200DMA: {row['Price/200DMA %']:.1f}%")
        if tech_str:
            lines.append(f"  Tech Data:    {' | '.join(tech_str)}")
            
        tot_score = row.get('Total Score', row['Score'])
        f_score = row.get('Fund. Score', 0)
        lines.append(f"  Total Score: {tot_score} (Tech: {row['Score']}, Fund: {f_score})  |  Filters: {row['Filters']}  |  Categories: {row['Categories']}")
        lines.append(f"  Type: {opp_type}")
        lines.append(f"  Signals: {row['Signals']}")
        lines.append(f"  Category Coverage: {row['Category List']}")

    # Category breakdown
    lines.append(f"\n{'=' * 70}")
    lines.append("  CATEGORY ANALYSIS")
    lines.append("=" * 70)

    for cat_name, cat_filters in SIGNAL_CATEGORIES.items():
        cat_tickers = set()
        for f in cat_filters:
            if f in results:
                cat_tickers.update(results[f]['Ticker'].tolist())
        if cat_tickers:
            lines.append(f"\n  {cat_name} ({len(cat_tickers)} stocks):")
            # Show top 5 by score from this category
            cat_scored = scored_df[scored_df['Ticker'].isin(cat_tickers)].head(5)
            for _, r in cat_scored.iterrows():
                lines.append(f"    {r['Ticker']:20s}  Score: {r['Score']:2d}  ({r['Filters']} filters)")

    # Multi-signal convergence
    lines.append(f"\n{'=' * 70}")
    lines.append("  CONVERGENCE ALERTS (3+ filter matches)")
    lines.append("=" * 70)

    convergence = scored_df[scored_df['Filters'] >= 3]
    if convergence.empty:
        lines.append("\n  No stocks found in 3+ filters.")
    else:
        for _, row in convergence.iterrows():
            opp_type = classify_opportunity(row)
            lines.append(f"\n  {row['Ticker']:15s}  [{opp_type:18s}]  Score: {row['Score']:2d}  |  {row['Signals']}")

    lines.append(f"\n{'=' * 70}")
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return '\n'.join(lines)


def generate_html_report(scored_df, results, data_date, top_n=15):
    """Generate a stylized HTML report of top opportunities."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Opportunity Analysis</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1000px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
            .summary-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #fafafa; }}
            .card-title {{ font-size: 1.2em; font-weight: bold; color: #2980b9; margin-bottom: 10px; display: flex; justify-content: space-between; }}
            .badge {{ background: #3498db; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }}
            .badge-score {{ background: #e74c3c; }}
            .signals {{ color: #7f8c8d; font-size: 0.9em; margin-top: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #34495e; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Stock Opportunity Analysis Report</h1>
                <p><strong>Data Date:</strong> {data_date} | <strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="summary-box">
                <p><strong>Filters Run:</strong> {len(results)} | <strong>Unique Stocks Found:</strong> {len(scored_df)}</p>
            </div>

            <h2>🏆 Top {top_n} Opportunities</h2>
    """

    top = scored_df.head(top_n)
    for idx, row in top.iterrows():
        opp_type = classify_opportunity(row)
        html += f"""
            <div class="card">
                <div class="card-title">
                    <span>#{idx+1} {row['Ticker']} <span class="badge">{opp_type}</span></span>
                    <span class="badge badge-score">Total Score: {row.get('Total Score', row['Score'])}</span>
                </div>
                <p style="margin:5px 0;">
                    <strong>Tech Score:</strong> {row['Score']} | 
                    <strong>Fund. Score:</strong> {row.get('Fund. Score', 0)} | 
                    <strong>Filters:</strong> {row['Filters']} | 
                    <strong>Categories:</strong> {row['Categories']}
                </p>
                <p style="margin:5px 0;">
                    <strong>Price:</strong> {row.get('Price', 'N/A')} | 
                    <strong>RSI:</strong> {row.get('RSI', 'N/A')} | 
                    <strong>Vol Ratio:</strong> {row.get('Vol Ratio', 'N/A')} | 
                    <strong>Price vs 200DMA:</strong> {row.get('Price/200DMA %', 'N/A')}%
                </p>
                <p style="margin:5px 0;">
                    <strong>Sector:</strong> {row.get('Sector', 'N/A')} | 
                    <strong>P/E:</strong> {row.get('P/E', 'N/A')} | 
                    <strong>ROE:</strong> {row.get('ROE %', 'N/A')}% | 
                    <strong>D/E:</strong> {row.get('D/E', 'N/A')} | 
                    <strong>Mkt Cap:</strong> {row.get('Mkt Cap', 'N/A')} Cr
                </p>
                <p style="margin:5px 0;"><strong>Signals:</strong> <span class="signals">{row['Signals']}</span></p>
            </div>
        """

    html += """
            <h2>📊 Category Analysis</h2>
    """

    for cat_name, cat_filters in SIGNAL_CATEGORIES.items():
        cat_tickers = set()
        for f in cat_filters:
            if f in results:
                cat_tickers.update(results[f]['Ticker'].tolist())
        if cat_tickers:
            html += f"<h3>{cat_name} ({len(cat_tickers)} stocks)</h3><table><tr><th>Ticker</th><th>Score</th><th>Filters</th></tr>"
            cat_scored = scored_df[scored_df['Ticker'].isin(cat_tickers)].head(5)
            for _, r in cat_scored.iterrows():
                html += f"<tr><td>{r['Ticker']}</td><td>{r['Score']}</td><td>{r['Filters']}</td></tr>"
            html += "</table>"

    html += """
        </div>
    </body>
    </html>
    """
    return html




def main():
    parser = argparse.ArgumentParser(description="Run stock analysis filters, generate opportunities, and export to Excel.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config JSON file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output Excel file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--html", default="opportunity_report.html", help="Output HTML report file")
    parser.add_argument("--txt", default="opportunity_report.txt", help="Output text report file")
    parser.add_argument("--top", type=int, default=15, help="Number of top opportunities to show in reports")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file '{args.config}' not found.")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Check data freshness and refresh if needed
    ensure_fresh_data()

    df, benchmark_df = load_data()
    latest_df = prepare_latest(df)

    data_date = df['Date'].max().strftime('%Y-%m-%d')
    print(f"Data date: {data_date}")
    print(f"Total tickers: {latest_df['Ticker'].nunique()}")
    print("-" * 60)

    # Run each enabled filter and collect into flat list
    all_rows = []
    summary_data = []
    results_dict = {}

    for key, (runner, sheet_name, needs_benchmark) in FILTER_REGISTRY.items():
        filter_cfg = cfg.get(key, {})
        if not filter_cfg.get("enabled", True):
            print(f"  [SKIP] {sheet_name:25s} - SKIPPED (disabled)")
            continue

        try:
            start = time.time()
            if needs_benchmark:
                result_df = runner(cfg, latest_df, df, benchmark_df, config_key=key)
            else:
                result_df = runner(cfg, latest_df, df, config_key=key)
            elapsed = time.time() - start

            count = len(result_df)
            summary_data.append({"Filter": sheet_name, "Stocks Found": count})
            print(f"  [ OK ] {sheet_name:25s} - {count:4d} stocks ({elapsed:.1f}s)")

            # Tag each row with the filter name
            if not result_df.empty:
                results_dict[sheet_name] = result_df
                tagged = result_df.copy()
                tagged.insert(0, 'Filter', sheet_name)
                all_rows.append(tagged)

        except Exception as e:
            print(f"  [FAIL] {sheet_name:25s} - ERROR: {e}")

    # Write to Excel
    if not all_rows:
        print("\nNo results to write.")
        return

    # Combine all rows into one flat DataFrame
    flat_df = pd.concat(all_rows, ignore_index=True)

    print(f"\n{'=' * 60}")
    print(f"Total rows: {len(flat_df)} ({flat_df['Ticker'].nunique()} unique tickers)")

    # --- Enrich with yfinance fundamentals ---
    flat_df = enrich_with_fundamentals(flat_df)

    # --- Score Stocks and Generate Reports ---
    print("\nScoring stocks for convergence...")
    scored_df = score_stocks(results_dict)

    scored_display = pd.DataFrame()
    if not scored_df.empty:
        # Build the Opportunities tab with fundamentals
        scored_display = scored_df.copy()
        metrics_cols = ['Ticker', 'Price', 'RSI', '200DMA Slope', '20DMA Slope', 'Vol Ratio', 'Price/200DMA %']
        metrics_df = build_display_df(latest_df, scored_df['Ticker'].tolist())
        available_cols = [c for c in metrics_cols if c in metrics_df.columns]
        scored_display = scored_display.merge(metrics_df[available_cols], on='Ticker', how='left')
        scored_display['Opportunity'] = scored_display.apply(classify_opportunity, axis=1)
        
        # Merge fundamentals from flat_df
        fund_cols = ['Ticker'] + [c for c in FUNDAMENTAL_FIELDS.values() if c in flat_df.columns]
        unique_funds = flat_df[fund_cols].drop_duplicates(subset=['Ticker'])
        scored_display = scored_display.merge(unique_funds, on='Ticker', how='left')

        # Fundamental Scoring
        def calc_fund_score(r):
            s = 0
            if pd.notna(r.get('P/E')) and 0 < r['P/E'] < 25: s += 2
            if pd.notna(r.get('ROE %')) and r['ROE %'] > 15: s += 2
            if pd.notna(r.get('D/E')) and r['D/E'] < 1.0: s += 1
            if pd.notna(r.get('Op Margin %')) and r['Op Margin %'] > 10: s += 1
            return s
            
        scored_display['Fund. Score'] = scored_display.apply(calc_fund_score, axis=1)
        scored_display['Total Score'] = scored_display['Score'] + scored_display['Fund. Score']
        
        # Sort by Total Score now
        scored_display = scored_display.sort_values(by=['Total Score', 'Score'], ascending=[False, False])

        # Reorder
        first_cols = ['Ticker', 'Opportunity', 'Total Score', 'Score', 'Fund. Score', 'Filters', 'Categories']
        remaining = [c for c in scored_display.columns if c not in first_cols]
        scored_display = scored_display[first_cols + remaining]

        # Generate Text Report
        txt_report = generate_report(scored_display, results_dict, data_date, args.top)
        with open(args.txt, 'w', encoding='utf-8') as f:
            f.write(txt_report)
        print(f"Text report saved to: {os.path.abspath(args.txt)}")

        # Generate HTML Report
        html_report = generate_html_report(scored_display, results_dict, data_date, args.top)
        with open(args.html, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML report saved to: {os.path.abspath(args.html)}")

    print(f"Writing to {args.output}...")

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        if not scored_display.empty:
            scored_display.to_excel(writer, sheet_name="Opportunities", index=False)
        
        # Main flat results sheet
        flat_df.to_excel(writer, sheet_name="All Results", index=False)

        # Summary sheet
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Done! Results saved to: {os.path.abspath(args.output)}")
    print(f"Data date: {data_date}")


if __name__ == "__main__":
    main()
