# %%

import nest_asyncio
nest_asyncio.apply()

import os
import pandas as pd


# %%
# load results

df_parts = []

for root,_, files in os.walk("./experiments"):
    for file in files:
        if file.endswith(".pkl"):
            df = pd.read_pickle(os.path.join(root, file))
            df_parts.append(df)

df = pd.concat(df_parts, ignore_index=True)
df.drop(columns=["generated_code", "fixed_code"], inplace=True)

# display results
df['model_name'].value_counts()

# %%

import matplotlib.pyplot as plt

# Specify the desired order of models
model_order = [
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:32b",

    "cogito:3b",
    "cogito:8b",
    "cogito:14b",
    "cogito:32b",

    "granite3.3:2b",
    "granite3.3",

    "llama3.2",
    "llama3.1",

    "phi4",
    # Add/remove model names as needed
]

# Add a new column 's' based on the specified conditions
df['score_final'] = df.apply(
    lambda row: 5 if row['score'] == 5 else
                4 if row['fix_score'] == 5 else
                1 if row['score'] == 1 else           
                1 if row['fix_score'] == 1 else 0,           
    axis=1
)

# Count each score per model_name and normalize to get percentages
score_counts = df.groupby('model_name')['score_final'].value_counts(normalize=True).unstack(fill_value=0)

# Ensure columns for all possible outcomes (0, 1, 4, 5)
for col in [0, 1, 4, 5]:
    if col not in score_counts.columns:
        score_counts[col] = 0
score_counts = score_counts[[0, 1, 4, 5]]

# Reindex to enforce the desired model order (drop missing if not present)
score_counts = score_counts.reindex(model_order).dropna(how="all")

# Plot stacked bar chart
ax = score_counts.plot(kind='bar', stacked=True, color=['red', 'orange', 'yellow', 'green'])
plt.ylabel('Percentage')
plt.title('Outcome Distribution by Model')
plt.legend(title='Score', labels=['Not Eval', 'Eval', 'Success on Fix', 'Success'])
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# %%
