"""
L5 — Communicating Results

- Calculates summary statistics
- Creates simple visualizations (saved to PNG)
- Writes a short textual summary (plain text)
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_summary(df):
    df['tip_pct'] = (df['tip']/df['total_bill']) * 100
    overall = df['tip_pct'].mean()
    median = df['tip_pct'].median()
    summary = {
        "overall_mean_tip_pct": round(overall, 2),
        "overall_median_tip_pct": round(median, 2),
        "observations": len(df)
    }
    return summary

def create_plots(df):
    sns.set_style("whitegrid")
    # Boxplot of tip_pct by day
    plt.figure(figsize=(8,5))
    sns.boxplot(x='day', y='tip_pct', data=df)
    plt.title("Tip percentage by day")
    plt.ylabel("Tip %")
    plt.savefig("tip_pct_by_day.png")
    plt.close()

    # Scatter: total_bill vs tip_pct
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='total_bill', y='tip_pct', hue='time', data=df)
    plt.title("Tip % vs Total bill")
    plt.savefig("tip_pct_vs_total_bill.png")
    plt.close()

def write_text_summary(summary):
    lines = [
        "Analysis Summary",
        "================",
        f"Observations: {summary['observations']}",
        f"Average tip %: {summary['overall_mean_tip_pct']}%",
        f"Median tip %: {summary['overall_median_tip_pct']}%",
        "",
        "Files output:",
        "- tip_pct_by_day.png",
        "- tip_pct_vs_total_bill.png"
    ]
    with open("analysis_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print("Wrote analysis_summary.txt and plot PNGs")

def main():
    df = sns.load_dataset("tips")
    summary = generate_summary(df)
    create_plots(df)
    write_text_summary(summary)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
