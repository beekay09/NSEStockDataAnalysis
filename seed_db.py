"""
One-time script to bulk-load existing CSV data into CockroachDB.
Run this once to seed the database, then never again.

Usage:
    python seed_db.py
"""
import pandas as pd
import db_utils

CSV_FILE = "nse_stock_data_with_metrics_v2.csv"


def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df['Date'] = pd.to_datetime(df['Date'])

    print(f"Loaded {len(df)} rows, {df['Ticker'].nunique()} unique tickers.")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    print("\nEnsuring table exists...")
    db_utils.ensure_table()

    print("\nFinding already seeded tickers to skip them...")
    engine = db_utils.get_engine()
    try:
        seeded_df = pd.read_sql('SELECT "Ticker", COUNT(*) as c FROM stock_metrics GROUP BY "Ticker"', engine)
        # Assuming 497 is the full history count per ticker (roughly 2 years)
        # Let's just skip tickers that have any data at all, and let App.py handle deltas,
        # OR skip tickers that have more than 100 rows.
        seeded_tickers = seeded_df[seeded_df['c'] > 400]['Ticker'].tolist()
        print(f"Skipping {len(seeded_tickers)} tickers that are already seeded.")
        df = df[~df['Ticker'].isin(seeded_tickers)]
        print(f"Remaining rows to seed: {len(df)}")
    except Exception as e:
        print(f"Could not filter seeded tickers (maybe table is empty): {e}")

    if df.empty:
        print("\n[OK] Database is already fully seeded!")
        return

    print("\nUpserting data to CockroachDB (this may take a few minutes)...")
    count = db_utils.upsert_metrics(df)

    if count > 0:
        print(f"\n[OK] Successfully seeded {count} rows into CockroachDB.")
    else:
        print("\n[X] Seeding failed. Check errors above.")

    # Verify
    max_date = db_utils.get_max_date()
    if max_date:
        print(f"DB max date: {max_date}")


if __name__ == "__main__":
    main()
