"""
L4 — Manipulating Data using Pandas and NumPy

Examples:
- filtering
- applying vectorized operations
- groupby + aggregation
- memory optimization (downcasting dtypes)
- pivot / reshape
"""
import pandas as pd
import seaborn as sns
import numpy as np

def optimize_dtypes(df):
    # Downcast numeric columns where appropriate
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    # Convert strings with few unique values to category
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    return df

def main():
    df = sns.load_dataset("tips")
    # Filtering: high spenders
    high_spenders = df[df['total_bill'] > 30].copy()
    print("High spenders shape:", high_spenders.shape)

    # Vectorized creation of a boolean column
    df['large_party'] = df['size'] >= 4

    # Groupby + aggregate: average tip and count per day
    agg = df.groupby('day').agg(
        avg_total_bill = ('total_bill', 'mean'),
        avg_tip = ('tip', 'mean'),
        orders = ('total_bill', 'size')
    ).round(2)
    print("\nAggregated stats by day:")
    print(agg)

    # Pivot table: average tip percentage per day x time
    df['tip_pct'] = (df['tip']/df['total_bill'])*100
    pivot = df.pivot_table(index='day', columns='time', values='tip_pct', aggfunc='mean')
    print("\nPivot table (tip_pct):")
    print(pivot.round(2))

    # Memory optimization
    before_mem = df.memory_usage(deep=True).sum()
    df_optimized = optimize_dtypes(df.copy())
    after_mem = df_optimized.memory_usage(deep=True).sum()
    print(f"\nMemory before: {before_mem/1024:.1f} KB, after: {after_mem/1024:.1f} KB")

    # Show first rows of optimized df
    print(df_optimized.head())

if __name__ == "__main__":
    main()
