import yfinance as yf
import pandas as pd
import datetime
import os

def fetch_stock_data(output_file="nse_stock_data.csv", start_date=None):
    # Read stocks from file
    # Read stocks from file
    stocks_file = "stocks.txt"
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            stocks = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(stocks)} stocks from {stocks_file}")
    else:
        print(f"{stocks_file} not found. Using static list of major NSE stocks.")
        stocks = [
            "HINDCOPPER", "TATACAP", "ENRIN", "POWERINDIA", "ABB", "ACC", "ADANIENSOL", "ADANIENT",
            "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBUJACEM", "APLAPOLLO", "APOLLOHOSP",
            "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
            "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BANKINDIA",
            "BDL", "BEL", "BHARATFORG", "BHARTIARTL", "BHARTIHEXA", "BHEL", "BIOCON", "BLUESTARCO",
            "BOSCHLTD", "BPCL", "BRITANNIA", "BSE", "CANBK", "CGPOWER", "CHOLAFIN", "CIPLA",
            "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "CUMMINSIND", "DABUR", "DEEPAKNTR", "DELHIVERY",
            "DIVISLAB", "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ELGIEQUIP", "EMAMILTD", "ESCORTS",
            "ETERNAL", "EXIDEIND", "FCL", "FEDERALBNK", "FIRSTCRY", "GAIL", "GILLETTE", "GLAXO",
            "GMRAIRPORT", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "HAL", "HAVELLS", "HCLTECH",
            "HDFCAMC", "HDFCBANK", "HEG", "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "HINDZINC",
            "HUDCO", "SAMMAANCAP", "ICICIBANK", "ICICIGI", "IDBI", "IDEA", "IDFCFIRSTB", "IGL",
            "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOB", "IOC",
            "IRB", "IREDA", "ITC", "JBCHEPHARM", "JINDALSTEL", "JKTYRE", "JSWENERGY", "JSWINFRA",
            "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KANSAINER", "KEI", "KIRLOSBROS", "KOTAKBANK",
            "KPITTECH", "LTF", "LICHSGFIN", "LICI", "LT", "LTTS", "LUPIN", "M&M", "M&MFIN",
            "MARICO", "MARUTI", "MAXHEALTH", "MAZDOCK", "MCX", "METROPOLIS", "MFSL", "MIDHANI",
            "MOTHERSON", "MPHASIS", "MRF", "MRPL", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NHPC",
            "NLCINDIA", "NMDC", "NTPC", "NYKAA", "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND",
            "PATANJALI", "POLICYBZR", "PERSISTENT", "PETRONET", "PFC", "PHOENIXLTD", "PIDILITIND",
            "PIIND", "PNB", "POLYCAB", "POWERGRID", "PREMIER", "PRESTIGE", "PVRINOX", "RAYMOND",
            "RBLBANK", "RECLTD", "RELIANCE", "RITES", "ROUTE", "RVNL", "SAIL", "SBICARD",
            "SBIN", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SJVN", "SOLARINDS",
            "SONACOMS", "SONATSOFTW", "SRF", "STARHEALTH", "SUNDARMFIN", "SUNPHARMA", "SUPREMEIND",
            "SUZLON", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAINVEST",
            "TMCV", "TMPV", "TATASTEEL", "TATATECH", "TCS", "TECHM", "TIINDIA", "TITAN",
            "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNIONBANK", "UPL",
            "VAIBHAVGBL", "VBL", "VEDL", "VGUARD", "VOLTAS", "WHIRLPOOL", "WIPRO", "YESBANK",
            "ZYDUSLIFE"
        ]


    # Remove duplicates if any
    stocks = list(set(stocks))

    # Add .NS suffix for NSE
    formatted_stocks = [f"{stock}.NS" for stock in stocks]

    print(f"Fetching data for {len(formatted_stocks)} stocks...")

    data_frames = []

    try:
        # Use start_date if provided (delta fetch), otherwise default to 2 years
        end_date = datetime.datetime.now()
        if start_date:
            print(f"Delta fetch: {start_date} to {end_date.date()}")
        else:
            start_date = end_date - datetime.timedelta(days=730)
            print(f"Full fetch: {start_date.date()} to {end_date.date()}")
        
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
