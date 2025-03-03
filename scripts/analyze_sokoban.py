import pandas as pd
from pathlib import Path

def analyze_parquet(filepath):
    """Analyze a parquet file and print basic statistics"""
    print(f"\nAnalyzing {filepath}")
    
    # Read the parquet file
    df = pd.read_parquet(filepath)
    
    # Basic info
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names and data types:")
    print(df.dtypes)
    
    # Sample data
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    # Define paths to parquet files
    data_dir = Path("RAGEN/data/sokoban")
    train_file = data_dir / "train.parquet"
    test_file = data_dir / "test.parquet"
    
    # Analyze both files
    analyze_parquet(train_file)
    analyze_parquet(test_file)
