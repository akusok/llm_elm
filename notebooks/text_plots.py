# %%

import nest_asyncio
nest_asyncio.apply()

import os
import pandas as pd


# %%
# load results

# df_parts = []

# for root,_, files in os.walk("./experiments"):
#     for file in files:
#         if file.endswith(".pkl"):
#             df = pd.read_pickle(os.path.join(root, file))
#             df_parts.append(df)

# df = pd.concat(df_parts, ignore_index=True)

df = pd.read_pickle("./experiments.pkl")
df.drop(columns=["generated_code", "fixed_code"], inplace=True)

# display results
df['model_name'].value_counts()

# %%

import matplotlib.pyplot as plt

# Specify the desired order of models
model_order = [
    "llama3.2",
    "llama3.1",

    "granite3.3:2b",
    "granite3.3",

    # "kwangsuklee/Phi4-mini-inst-Q4_K_M",
    "phi4",

    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:32b",

    "cogito:3b",
    "cogito:8b",
    "cogito:14b",
    "cogito:32b",
    # Add/remove model names as needed

    # "exaone-deep",
    # "deepseek-r1:8b",
    "deepseek-r1:8b",
    # "gemma3:12b-it-qat",
    "command-r7b",
    "opencoder",
    "cogito:8b",
    "llama3.1",
    # "yandex/YandexGPT-5-Lite-8B-instruct-GGUF",
    "tulu3",
    "falcon3",
    "granite3.3",
    "marco-o1",
    "qwen2.5-coder:7b",
    "exaone3.5",    
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
# convert to percentage
score_counts = (score_counts * 100).round(2)

# patch: set 'deepseek-r1:8b' to 0
score_counts.loc['deepseek-r1:8b'] = 0

# Ensure columns for all possible outcomes (0, 1, 4, 5)
for col in [0, 1, 4, 5]:
    if col not in score_counts.columns:
        score_counts[col] = 0
score_counts = score_counts[[0, 1, 4, 5]]

# Reindex to enforce the desired model order (drop missing if not present)
score_counts = score_counts.reindex(model_order).dropna(how="all")

# rename columns for clarity
score_counts.rename(columns={0: 'Fail', 1: 'Eval', 4: 'Success on Fix', 5: 'Success'}, inplace=True)
# rename some index names for clarity
score_counts.rename(index={
    'yandex/YandexGPT-5-Lite-8B-instruct-GGUF': 'yandex:8b',
    'kwangsuklee/Phi4-mini-inst-Q4_K_M': 'phi4-mini:4b',
    'llama3.2': 'llama3.2:3b',
    'llama3.1': 'llama3.1:8b',
    'granite3.3': 'granite3.3:8b',
    'phi4': 'phi4:14b',
    'opencoder': 'opencoder:8b',
    'tulu3': 'tulu3:8b',
    'falcon3': 'falcon3:7b',
    'marco-o1': 'marco-o1:7b',
    'exaone3.5': 'exaone3.5:8b',
    'deepseek-r1:8b': ' ',
}, inplace=True)

# Plot stacked bar chart
ax = score_counts.plot(kind='bar', stacked=True, color=['red', 'orange', 'yellow', 'green'], legend=True, figsize=(13, 4))
plt.ylabel('outcomes')
# plt.title('Outcome Distribution by Model')
# plt.legend(title='Score', labels=['Not Eval', 'Eval', 'Success on Fix', 'Success'])
# hide grid behind bars
ax.set_axisbelow(True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 100)
plt.tight_layout()
# hide x label
ax.set_xlabel('')
plt.show()

# save to png, no borders
fig = ax.get_figure()
fig.savefig(
    "teaser.png",
    bbox_inches='tight',
    pad_inches=0,
    dpi=300,
    transparent=True
)

# %%
