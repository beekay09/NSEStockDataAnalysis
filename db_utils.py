"""
Database utility module for CockroachDB integration.
Handles all DB operations: connect, read, upsert.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Column list matching the stock_metrics table schema
TABLE_COLUMNS = [
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "3DMA", "20DMA", "100DMA", "200DMA",
    "3VWMA", "20VWMA", "100VWMA", "200VWMA",
    "HA_Close", "HA_Open",
    "3DMA_SLOPE", "20DMA_SLOPE", "100DMA_SLOPE", "200DMA_SLOPE", "RSI_14_SLOPE"
]

TABLE_NAME = "stock_metrics"

_engine = None


def get_engine():
    """Creates and caches a SQLAlchemy engine from DATABASE_URL env var."""
    global _engine
    if _engine is None:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Set it to your CockroachDB connection string."
            )
        # CockroachDB requires the cockroachdb:// dialect for proper version parsing
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "cockroachdb://", 1)
        _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine


def ensure_table():
    """Creates the stock_metrics table if it doesn't exist."""
    engine = get_engine()
    create_sql = text("""
        CREATE TABLE IF NOT EXISTS stock_metrics (
            "Date" DATE,
            "Ticker" VARCHAR(50),
            "Open" DOUBLE PRECISION,
            "High" DOUBLE PRECISION,
            "Low" DOUBLE PRECISION,
            "Close" DOUBLE PRECISION,
            "Volume" BIGINT,
            "RSI_14" DOUBLE PRECISION,
            "3DMA" DOUBLE PRECISION,
            "20DMA" DOUBLE PRECISION,
            "100DMA" DOUBLE PRECISION,
            "200DMA" DOUBLE PRECISION,
            "3VWMA" DOUBLE PRECISION,
            "20VWMA" DOUBLE PRECISION,
            "100VWMA" DOUBLE PRECISION,
            "200VWMA" DOUBLE PRECISION,
            "HA_Close" DOUBLE PRECISION,
            "HA_Open" DOUBLE PRECISION,
            "3DMA_SLOPE" DOUBLE PRECISION,
            "20DMA_SLOPE" DOUBLE PRECISION,
            "100DMA_SLOPE" DOUBLE PRECISION,
            "200DMA_SLOPE" DOUBLE PRECISION,
            "RSI_14_SLOPE" DOUBLE PRECISION,
            PRIMARY KEY ("Date", "Ticker")
        );
    """)
    with engine.connect() as conn:
        conn.execute(create_sql)
        conn.commit()
    print("Table 'stock_metrics' ensured.")


def read_all_from_db():
    """Reads the entire stock_metrics table into a pandas DataFrame."""
    engine = get_engine()
    try:
        df = pd.read_sql_table(TABLE_NAME, con=engine)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
        print(f"Read {len(df)} rows from DB.")
        return df
    except Exception as e:
        print(f"Error reading from DB: {e}")
        return pd.DataFrame()


def get_max_date():
    """Returns the maximum Date in the stock_metrics table, or None if empty.
    
    Raises an exception on connection/query errors so callers can distinguish
    between an empty table (None) and a DB failure (exception).
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(f'SELECT MAX("Date") FROM {TABLE_NAME}'))
        row = result.fetchone()
        if row and row[0]:
            return pd.Timestamp(row[0])
    return None


def upsert_metrics(df):
    """
    Bulk upsert a DataFrame into stock_metrics using
    INSERT ... ON CONFLICT (Date, Ticker) DO UPDATE.
    
    Only includes columns that exist in TABLE_COLUMNS.
    """
    if df.empty:
        print("No data to upsert.")
        return 0

    engine = get_engine()
    ensure_table()

    # Filter to only columns that match the table schema
    available_cols = [c for c in TABLE_COLUMNS if c in df.columns]
    upsert_df = df[available_cols].copy()

    # Ensure Date is date-only (no time component)
    if 'Date' in upsert_df.columns:
        upsert_df['Date'] = pd.to_datetime(upsert_df['Date']).dt.date

    # Drop rows with null PK values
    upsert_df = upsert_df.dropna(subset=['Date', 'Ticker'])

    # Build the UPSERT SQL
    col_names = ', '.join([f'"{c}"' for c in available_cols])
    placeholders = ', '.join([f':{c}' for c in available_cols])
    update_set = ', '.join([
        f'"{c}" = excluded."{c}"'
        for c in available_cols
        if c not in ('Date', 'Ticker')
    ])

    upsert_sql = text(f"""
        INSERT INTO {TABLE_NAME} ({col_names})
        VALUES ({placeholders})
        ON CONFLICT ("Date", "Ticker") DO UPDATE SET {update_set}
    """)

    # Execute in smaller batches to prevent CockroachDB connection timeouts
    batch_size = 100
    total_rows = len(upsert_df)
    upserted = 0

    try:
        for start in range(0, total_rows, batch_size):
            batch = upsert_df.iloc[start:start + batch_size]
            records = batch.to_dict(orient='records')
            # Open a new connection per batch to avoid long-running transaction timeouts
            with engine.connect() as conn:
                conn.execute(upsert_sql, records)
                conn.commit()
            upserted += len(records)
            print(f"  Upserted {upserted}/{total_rows} rows...", end='\r')
        print(f"\nSuccessfully upserted {upserted} rows to DB.")
        return upserted
    except SQLAlchemyError as e:
        print(f"\nError upserting to DB: {e}")
        return 0
