# %%

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import nest_asyncio
nest_asyncio.apply()

# %%
# load mnist dataset

import os
import uuid
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sys

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %%
# evaluate generated code on mnist

def eval_code(code:str) -> tuple[int,str]:
    """
    Evaluate the generated code on MNIST dataset.

    Returns:
        code does not eval: 0
        eval but has errors: 1
        runs successfully: 5
    """

    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()

    # Create a local namespace to execute the code
    local_namespace = {}

    try:
        # Execute the code in the local namespace
        exec(code, local_namespace)
    except Exception as e:
        # print(f"Error executing code: {e}")
        return 0, "Code does not eval"

    try:
        # Extract the function from the local namespace
        train_hpelm_mnist = local_namespace['train_hpelm_mnist']

        # Evaluate the function on the MNIST dataset
        accuracy = train_hpelm_mnist(X_train, y_train, X_test, y_test)
        # print(f"Model accuracy: {accuracy}")
        return 5, str(accuracy)
    
    except Exception as e:
        # print(f"Error executing function: {e}")
        return 1, str(e)


# %%
# setup ollama

# model_name='qwen2.5-coder:32b'
# model_name='qwen2.5-coder:14b'
# model_name='qwen2.5-coder:7b'
# model_name='qwen2.5-coder:3b'

# model_name='granite3.3'

# model_name='cogito:32b'
# model_name='cogito:14b'
# model_name='cogito:7b'
# model_name='cogito:3b'

# model_name='llama3.1'
# model_name='llama3.2'

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    print("Please provide a model name as the first terminal parameter.")
    sys.exit(1)


ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

simple_agent = Agent(ollama_model)


# %%
# generate code

prompt = """
    Write a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` that converts MNIST targets 
    to one-hot encoding, trains an ELM model, evaluates its performance, and returns test accuracy.
    Use L2 regularization to avoid overfitting.
    Return only function code and imports, remove examples or other python code after the function.

    - Input values: X_train and X_test have shape (num_samples, 784), y_train and y_test are 1D arrays of shape (num_samples,)
    - Task is a 10-class classification problem
"""

# Append the ELM implementation to the prompt
with open('../doc/hpelm_doc_alt.md', 'r') as file:
    full_prompt = prompt + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()

# # response = py_agent.run_sync(full_prompt)
# response = simple_agent.run_sync(full_prompt)
# generated_code = response.data

# print(generated_code)
# print(response.usage())


# %%
# evaluate generated code on mnist

# score, msg = eval_code(generated_code)
# print(f"Score: {score}, Message: {msg}")

# try fixing code with errors
fix_request = """
Fix the code to be run by exec(code) in Python.: {generated_code}
Return only the code, no explanation.

Error: {msg}
"""

# if score == 1:
#     rq = fix_request.format(generated_code, msg)
#     fixed_code = simple_agent.run_sync(rq).data
#     score, msg = eval_code(fixed_code)
#     print("After fixing:")
#     print(f"Score: {score}, Message: {msg}")


# %%
# run experiment multiple times and gather statistics

experimental_results = []

for i in range(10):
    print()
    print(f"Experiment {i+1}")

    generated_code = simple_agent.run_sync(full_prompt).data
    score, msg = eval_code(generated_code)

    if score == 1:
        rq = fix_request.format(generated_code=generated_code, msg=msg)
        fixed_code = simple_agent.run_sync(rq).data
        fix_score, fix_msg = eval_code(fixed_code)

    experimental_results.append({
        "model_name": model_name,
        "experiment": i,
        "score": score,
        "msg": msg,
        "fix_score": fix_score if score == 1 else 0,
        "fix_msg": fix_msg if score == 1 else "",
        "generated_code": generated_code,
        "fixed_code": fixed_code if score == 1 else "",
    })
 
# save results as pandas dataframe to a TCV file
pd.DataFrame(experimental_results).to_pickle(
    f"./experiments/mnist_experiments_{uuid.uuid4()}.pkl"
)


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
df.head(1000)

# %%
