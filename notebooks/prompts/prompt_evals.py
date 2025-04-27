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

model_name='qwen2.5-coder:7b'


ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

simple_agent = Agent(ollama_model)


# %%
# generate code

prompt1 = """
    Write a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` that converts MNIST targets 
    to one-hot encoding, trains an ELM model, evaluates its performance, and returns test accuracy.
    Use L2 regularization to avoid overfitting.
    Return only function code and imports, remove examples or other python code after the function.

    - Input values: X_train and X_test have shape (num_samples, 784), y_train and y_test are 1D arrays of shape (num_samples,)
    - Task is a 10-class classification problem
"""

prompt2 = """
    Write a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` that 
    trains an ELM model on MNIST dataset, evaluates its performance, and returns test accuracy.
    Use L2 regularization to avoid overfitting.
    Return only function code and imports, remove examples or other python code after the function.
"""

prompt3 = """
    Train an ELM model on the MNIST dataset using the `hpelm` library, and return the test accuracy. 
    Function should be `train_hpelm_mnist(X_train, y_train, X_test, y_test)`.
    Use L2 regularization to avoid overfitting.
"""

prompt4 = """
    Write a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` 
    that trains on MNIST dataset and returns accuracy.
"""



# Append the ELM implementation to the prompt
with open('../../doc/hpelm_doc_alt.md', 'r') as file:
    full_prompt1 = prompt1 + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()
    full_prompt2 = prompt2 + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()
    full_prompt3 = prompt3 + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()
    full_prompt4 = prompt4 + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()

full_prompts = [full_prompt1, full_prompt2, full_prompt3, full_prompt4]

fix_request = """
Fix the code to be run by exec(code) in Python.: {generated_code}
Return only the code, no explanation.

Error: {msg}
"""

# %%
# run experiment multiple times and gather statistics

experimental_results = []
fname_out = f"./prompt_{model_name.replace("/", "_")}_{uuid.uuid4()}.pkl"

n_attempts = 10 if len(sys.argv) < 2 else int(sys.argv[2])

for i in range(n_attempts):
    print()
    print(f"Experiment {i+1}")

    for j in range(4):
        generated_code = simple_agent.run_sync(full_prompts[j]).data
        score, msg = eval_code(generated_code)

        if score == 1:
            print("Code has errors, trying to fix it...")
            rq = fix_request.format(generated_code=generated_code, msg=msg)
            fixed_code = simple_agent.run_sync(rq).data
            fix_score, fix_msg = eval_code(fixed_code)

        experimental_results.append({
            "model_name": model_name,
            "prompt": j,
            "experiment": i,
            "score": score,
            "msg": msg,
            "fix_score": fix_score if score == 1 else 0,
            "fix_msg": fix_msg if score == 1 else "",
            "generated_code": generated_code,
            "fixed_code": fixed_code if score == 1 else "",
        })
    
        # save results as pandas dataframe to a TCV file
        pd.DataFrame(experimental_results).to_pickle(fname_out)
