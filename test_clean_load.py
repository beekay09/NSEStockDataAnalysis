import App
import pandas as pd

try:
    df, benchmark_df = App.load_data()
    print(f"Dataframe shape: {df.shape}")
    if not df.empty:
        print("First 5 rows:")
        print(df.head())
        print("Columns:", df.columns.tolist())
    else:
        print("Dataframe is empty!")
    
    print(f"Benchmark shape: {benchmark_df.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
