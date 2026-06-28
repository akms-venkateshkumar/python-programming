"""
data_analysis_process.py

A short script that demonstrates the typical data analysis process:
- load
- explore / inspect
- clean / transform
- analyze
- communicate

Uses seaborn's "tips" dataset for demonstration.
"""
import pandas as pd
import numpy as np
import seaborn as sns

def load_data():
    df = sns.load_dataset("tips")
    print("Loaded dataset with shape:", df.shape)
    return df

def explore_data(df):
    print("\n--- HEAD ---")
    print(df.head())
    print("\n--- INFO ---")
    print(df.info())
    print("\n--- DESCRIBE ---")
    print(df.describe(include='all'))

def clean_data(df):
    # Example cleaning: standardize column names, handle missing values
    df = df.rename(columns=str.strip).copy()
    # There are no missing values in tips, but show approach:
    df['tip'] = df['tip'].fillna(df['tip'].median())
    # Ensure categorical dtypes
    for col in ['sex', 'smoker', 'day', 'time']:
        df[col] = df[col].astype('category')
    return df

def analyze(df):
    # Add a derived column: tip percentage
    df['tip_pct'] = (df['tip'] / df['total_bill']) * 100
    grouped = df.groupby(['day', 'time'])['tip_pct'].mean().unstack()
    print("\nAverage tip percentage by day and time:")
    print(grouped.round(2))
    return df, grouped

def communicate(df, grouped):
    # Simple textual summary
    overall_tip_pct = df['tip_pct'].mean()
    print(f"\nOverall average tip percentage: {overall_tip_pct:.2f}%")
    # Save summarized table to CSV for stakeholders
    grouped.to_csv("avg_tip_pct_by_day_time.csv")
    print("Saved avg_tip_pct_by_day_time.csv")

def main():
    df = load_data()
    explore_data(df)
    df = clean_data(df)
    df, grouped = analyze(df)
    communicate(df, grouped)

if __name__ == "__main__":
    main()
