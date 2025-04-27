# %%

import nest_asyncio
nest_asyncio.apply()

import os
import pandas as pd

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model_name='qwen2.5-coder:7b'


ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

simple_agent = Agent(ollama_model)

from pydantic import BaseModel
from pydantic_ai.exceptions import UnexpectedModelBehavior

class PythonFunctionModel(BaseModel):
    python_code: str  # Python code of the function

# Create a new agent with the formatted output model
formatted_agent = Agent(
    ollama_model,
    output_type=PythonFunctionModel
)


df_parts = []

for root,_, files in os.walk("."):
    for file in files:
        if file.endswith(".pkl"):
            df = pd.read_pickle(os.path.join(root, file))
            df_parts.append(df)

df = pd.concat(df_parts, ignore_index=True)

df.drop(columns=["generated_code", "fixed_code"], inplace=True)

# display results
df['format'].value_counts()

# %%

df_parts = []

for root,_, files in os.walk("."):
    for file in files:
        if file.endswith(".pkl"):
            df = pd.read_pickle(os.path.join(root, file))
            df_parts.append(df)

df = pd.concat(df_parts, ignore_index=True)
df = df[df['format'] == "pydantic"][['generated_code']]

from IPython.display import display
display(df.head(20).style.set_caption("Sample of DataFrame").background_gradient(cmap="YlGnBu"))

# %%

import matplotlib.pyplot as plt


# Add a new column 's' based on the specified conditions
df['score_final'] = df.apply(
    lambda row: 5 if row['score'] == 5 else
                4 if row['fix_score'] == 5 else
                1 if row['score'] == 1 else           
                1 if row['fix_score'] == 1 else 0,           
    axis=1
)

# Count each score per model_name and normalize to get percentages
score_counts = df.groupby('format')['score_final'].value_counts(normalize=True).unstack(fill_value=0)
# convert to percentage
score_counts = (score_counts * 100).round(2)

# Ensure columns for all possible outcomes (0, 1, 4, 5)
for col in [0, 1, 4, 5]:
    if col not in score_counts.columns:
        score_counts[col] = 0
score_counts = score_counts[[0, 1, 4, 5]]

# Reindex to enforce the desired model order (drop missing if not present)
# score_counts = score_counts.reindex(model_order).dropna(how="all")

# Plot 4 pie charts, one for each prompt
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

score_labels = {0: "Fail", 1: "Eval", 4: "Fixed", 5: "Pass"}
colors = {0: "red", 1: "orange", 4: "yellow", 5: "green"}

titles = [
    "Full prompt, all details",
    "Missing data shape",
]

for i, (format, row) in enumerate(score_counts.iterrows()):
    ax = axes[i]
    values = row.values
    labels = [score_labels[k] for k in score_counts.columns]
    if i != 0:
        labels = None
    pie_colors = [colors[k] for k in score_counts.columns]
    ax.pie(
        values,
        # labels=labels,
        autopct='%1.0f%%',
        startangle=90,
        colors=pie_colors,
        textprops={'fontsize': 10}
    )
    ax.set_title(format, fontsize=10, pad=0)  # Adjusted pad to bring title closer
    ax.axis('equal')

# Add a separate legend
fig.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[k], markersize=10) for k in score_labels],
    labels=[score_labels[k] for k in score_labels],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=len(score_labels),
    fontsize=10
)


plt.tight_layout()
plt.show()

# %%

# save to png, no borders
fig = ax.get_figure()
fig.savefig(
    "format.png",
    bbox_inches='tight',
    pad_inches=0,
    dpi=300,
    transparent=True
)

# %%
