"""
L3 — Exploring and Inspecting Data
Shows common pandas commands to inspect data properties and answer starter questions.
"""
import pandas as pd
import seaborn as sns
import numpy as np

def main():
    df = sns.load_dataset("tips")
    # Basic properties
    print("Shape:", df.shape)
    print("\nColumns and dtypes:")
    print(df.dtypes)

    # Show missing values per column
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Head & tail
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 3 rows:")
    print(df.tail(3))

    # Statistical summary for numeric columns
    print("\nNumeric summary:")
    print(df.describe())

    # Categorical summaries
    for col in ['sex', 'smoker', 'day', 'time']:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

    # Example exploratory question: which day has the highest median total bill?
    medians_by_day = df.groupby('day')['total_bill'].median().sort_values(ascending=False)
    print("\nMedian total bill by day:")
    print(medians_by_day)

if __name__ == "__main__":
    main()
