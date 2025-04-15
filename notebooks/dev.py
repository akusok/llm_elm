# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: jupytext_format_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Jupytext notebook

# %%
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# %%
# fix Pydantic in jupyter

import nest_asyncio
nest_asyncio.apply()

# %%

ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

# %%

agent = Agent(  
    ollama_model,
    system_prompt='Be concise, reply with one sentence.',
)

# %%

result = agent.run_sync('Where does "hello world" come from?')  
print(result.data)

# %%

print("done")

# %%
