# %% [markdown]
# # L2 — Jupyter Notebook Example
# This example shows how to structure analysis in notebook cells.
# Paste the code cells into a notebook or open in VS Code and use the interactive runner.

# %% [markdown]
# ## Cell 1: Imports and dataset load

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")
df = sns.load_dataset("tips")
print("Dataset shape:", df.shape)

# %% [markdown]
# ## Cell 2: Quick peek

# %%
df.head()

# %% [markdown]
# ## Cell 3: Inline plot (visualization cell)

# %%
plt.figure(figsize=(8,5))
sns.histplot(df['total_bill'], bins=20, kde=True)
plt.title("Total bill distribution")
plt.xlabel("Total bill ($)")
plt.show()

# %% [markdown]
# ## Cell 4: Derived metric and table

# %%
df['tip_pct'] = (df['tip']/df['total_bill']) * 100
df.groupby('day')['tip_pct'].agg(['mean','median','count']).round(2)

# %% [markdown]
# ## Cell 5: Save results and close (export)

# %%
df.to_csv("tips_with_tip_pct.csv", index=False)
print("Saved tips_with_tip_pct.csv")
