-- price_history: raw daily OHLCV, one row per (symbol, date)
CREATE TABLE price_history (
  symbol TEXT NOT NULL REFERENCES stocks(symbol),
  date TEXT NOT NULL,          -- ISO date
  open REAL, high REAL, low REAL, close REAL, volume INTEGER,
  PRIMARY KEY (symbol, date)
);
CREATE INDEX idx_price_symbol_date ON price_history(symbol, date);

-- benchmark_price_history: index data (Nifty 50 etc.) needed for beta / 
-- relative strength -- separate table since it's not a "stock"
CREATE TABLE benchmark_price_history (
  benchmark TEXT NOT NULL,     -- e.g. 'NIFTY50'
  date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume INTEGER,
  PRIMARY KEY (benchmark, date)
);

-- technical_indicators: one row per (symbol, date), all derived metrics
CREATE TABLE technical_indicators (
  symbol TEXT NOT NULL REFERENCES stocks(symbol),
  date TEXT NOT NULL,
  rsi_14 REAL,
  dma_3 REAL, dma_20 REAL, dma_100 REAL, dma_200 REAL,
  vwma_3 REAL, vwma_20 REAL, vwma_100 REAL, vwma_200 REAL,
  dma_3_slope REAL, dma_20_slope REAL, dma_100_slope REAL, dma_200_slope REAL,
  rsi_14_slope REAL,
  ha_open REAL, ha_high REAL, ha_low REAL, ha_close REAL,
  bb_sma REAL, bb_upper REAL, bb_lower REAL,
  macd REAL, macd_signal REAL, macd_hist REAL,
  adx REAL, plus_di REAL, minus_di REAL,
  daily_return REAL,
  beta_vs_market REAL,         -- vs equal-weighted synthetic index of fetched universe
  beta_vs_nifty50 REAL,        -- vs actual benchmark, once fetched
  rs_ratio REAL, rs_ma REAL, rs_slope REAL,   -- vs NIFTY50
  computed_at TEXT NOT NULL,
  PRIMARY KEY (symbol, date)
);
CREATE INDEX idx_tech_symbol_date ON technical_indicators(symbol, date);
