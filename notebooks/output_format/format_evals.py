# %%


from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import nest_asyncio
nest_asyncio.apply()

# %%
# load mnist dataset

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

# model_name='qwen2.5-coder:7b'
model_name='granite3.3'


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
    output_type=PythonFunctionModel,
    retries=3,
)

# %%


# %%


# ollama client
py_agent = Agent(ollama_model)

prompt = """
    Write a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` that converts MNIST targets 
    to one-hot encoding, trains an ELM model, evaluates its performance, and returns test accuracy.
    Use L2 regularization to avoid overfitting.
    Return only function code and imports, remove examples or other python code after the function.

    - Input values: X_train and X_test have shape (num_samples, 784), y_train and y_test are 1D arrays of shape (num_samples,)
    - Task is a 10-class classification problem
"""

# Load the content of 'hpelm_info.md' file
with open('../../doc/hpelm_doc_alt.md', 'r') as file:
    full_prompt = prompt + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()

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

    # baseline
    generated_code = simple_agent.run_sync(full_prompt).data
    score, msg = eval_code(generated_code)

    if score == 1:
        print("Code has errors, trying to fix it...")
        rq = fix_request.format(generated_code=generated_code, msg=msg)
        fixed_code = simple_agent.run_sync(rq).data
        fix_score, fix_msg = eval_code(fixed_code)

    experimental_results.append({
        "model_name": model_name,
        "format": "plain text",
        "experiment": i,
        "score": score,
        "msg": msg,
        "fix_score": fix_score if score == 1 else 0,
        "fix_msg": fix_msg if score == 1 else "",
        "generated_code": generated_code,
        "fixed_code": fixed_code if score == 1 else "",
    })

    try:
        generated_code = formatted_agent.run_sync(full_prompt).output.python_code
    except UnexpectedModelBehavior as e:
        generated_code = str(e)
    score, msg = eval_code(generated_code)

    experimental_results.append({
        "model_name": model_name,
        "format": "pydantic",
        "experiment": i,
        "score": score,
        "msg": msg,
        "fix_score": 0,
        "fix_msg": "",
        "generated_code": generated_code,
        "fixed_code": "",
    })

    # save results as pandas dataframe to a TCV file
    pd.DataFrame(experimental_results).to_pickle(fname_out)

# %%

