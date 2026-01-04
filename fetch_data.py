import yfinance as yf
import pandas as pd
import datetime
import os

def fetch_stock_data(output_file="nse_stock_data.csv"):
    # Read stocks from file
    stocks_file = "stocks.txt"
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            stocks = [line.strip() for line in f if line.strip()]
    else:
        print(f"Error: {stocks_file} not found.")
        return

    # Remove duplicates if any
    stocks = list(set(stocks))

    # Add .NS suffix for NSE
    formatted_stocks = [f"{stock}.NS" for stock in stocks]

    print(f"Fetching data for {len(formatted_stocks)} stocks...")

    data_frames = []

    try:
        # Period '2y' is approximately 730 days.
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=730)
        
        # Download data
        # threads=True uses threading for faster download
        df = yf.download(formatted_stocks, start=start_date, end=end_date, group_by='ticker', threads=True)
        
        # Flatten the MultiIndex columns if necessary
        # yfinance v0.2.40+ often returns a MultiIndex even for single tickers
        # The user suggested using droplevel to handle this overhead.
        
        stacked_data = []
        
        # Check if df columns are MultiIndex (which they should be for multiple stocks)
        if isinstance(df.columns, pd.MultiIndex):
            for ticker in formatted_stocks:
                try:
                    # Extract dataframe for this ticker
                    # If columns are (Price, Ticker), xs(ticker, level=1) is cleanest
                    # If columns are (Ticker, Price), df[ticker] works.
                    # We ensure we flatten any remaining MultiIndex as suggested.
                    if 'Ticker' in df.columns.names:
                        stock_df = df.xs(ticker, axis=1, level='Ticker').copy()
                    else:
                        stock_df = df[ticker].copy()
                        if isinstance(stock_df.columns, pd.MultiIndex):
                            stock_df = stock_df.droplevel(level=1, axis=1)
                    
                    # Drop rows with all NaNs (meaning no data for that date for this stock)
                    stock_df.dropna(how='all', inplace=True)
                    
                    if not stock_df.empty:
                        # Add Ticker column
                        stock_df['Ticker'] = ticker.replace('.NS', '')
                        stock_df.reset_index(inplace=True)
                        stacked_data.append(stock_df)
                except KeyError:
                    print(f"No data found for {ticker} in the batch download.")
                    continue
        else:
            # Fallback if somehow single ticker or structure is different
            print("Unexpected data format. Proceeding with flat structure check.")
            df['Ticker'] = formatted_stocks[0].replace('.NS', '') 
            df.reset_index(inplace=True)
            stacked_data.append(df)
            
        if stacked_data:
            final_df = pd.concat(stacked_data, ignore_index=True)
            
            # Reorder columns to have Date and Ticker first
            cols = ['Date', 'Ticker'] + [c for c in final_df.columns if c not in ['Date', 'Ticker']]
            final_df = final_df[cols]
            
            final_df.to_csv(output_file, index=False)
            print(f"Successfully saved data to {output_file}")
            print(f"Total records: {len(final_df)}")
        else:
            print("No data fetched.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_stock_data()
