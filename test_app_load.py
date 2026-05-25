import App
import time

print("Testing App.load_data()...")
start_time = time.time()
try:
    df, benchmark_df = App.load_data()
    print(f"Success! Dataframe shape: {df.shape}")
    print(f"Benchmark shape: {benchmark_df.shape}")
except Exception as e:
    print(f"Failed: {e}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")
