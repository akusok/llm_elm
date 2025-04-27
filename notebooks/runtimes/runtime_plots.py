# %%
import pandas as pd

# Load the DataFrame from the pickle file
df1 = pd.read_pickle("runtimes_results.pkl")
df2 = pd.read_pickle("runtimes_results_2.pkl")
df3 = pd.read_pickle("runtimes_results_3.pkl")
df = pd.concat([df1, df2, df3], ignore_index=True)

# Display the first few rows of the DataFrame
print(df.head(99))

# drop first row
df = df[df["j"] != 0]
# %%
import matplotlib.pyplot as plt

# Group by model_name and calculate the mean of runtime_sec, first_token_sec, and num_tokens
result = df.groupby("model_name")[["runtime_sec", "num_tokens"]].mean()
result = result.reindex(["llama3.2", "qwen2.5-coder:7b", "llama3.1", 
                         "phi4", "qwen2.5-coder:32b", "marco-o1", "deepseek-r1:8b"]).dropna(how="all")

# rename the index
result.rename(index={
    "phi4": "phi4:14b",
    "llama3.2": "llama3.2:3b",
    "llama3.1": "llama3.1:8b",
    "marco-o1": "marco-o1:7b"
}, inplace=True)

# Create a figure and axis objects
fig, ax1 = plt.subplots(figsize=(8, 4))

bar_width = 0.35
x = range(len(result.index))

# Plot runtime_sec and first_token_sec on the left y-axis
ax1.bar(x, result["runtime_sec"], width=bar_width, label="Runtime (sec)", alpha=0.7)
ax1.set_ylabel("Time (sec)")
# ax1.set_xlabel("Model Name")
ax1.set_xticks([i + bar_width/2 for i in x])
ax1.set_xticklabels(result.index, rotation=45)

# Create a second y-axis for num_tokens
ax2 = ax1.twinx()
ax2.bar([i + 1 * bar_width for i in x], result["num_tokens"], width=bar_width, color="tab:green", label="Num Tokens", alpha=0.7)
ax2.set_ylabel("Num Tokens")

# Combine legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

# plt.title("Average Runtime, First Token Time, and Num Tokens by Model")
plt.tight_layout()
plt.show()
# %%

# save to png, no borders
fig = ax1.get_figure()
fig.savefig(
    "runtime.png",
    bbox_inches='tight',
    pad_inches=0,
    dpi=300,
    transparent=True
)
# %%
