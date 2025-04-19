# %%
from pydantic import BaseModel, field_validator
import numpy as np

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import nest_asyncio
nest_asyncio.apply()


# %%
# setup ollama

ollama_model = OpenAIModel(
    # model_name='qwen2.5-coder:14b',
    # model_name='qwen2.5-coder:7b',
    # model_name='qwen2.5-coder:3b',

    model_name='granite3.3',

    # model_name='cogito:14b',
    # model_name='llama3.1',
    # model_name='llama3.2',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

simple_agent = Agent(ollama_model)


# %%
# structured output

class CityLocation(BaseModel):
    city: str
    country: str

structured_agent = Agent(ollama_model, result_type=CityLocation)

result = structured_agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())

#> Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)

# %%
# generate code

class PythonFunctionModel(BaseModel):
    code: str # code of the function train_hpelm_mnist(X_train, y_train, X_test, y_test)

# ollama client
py_agent = Agent(ollama_model, result_type=PythonFunctionModel, retries=1)

prompt = """
    Write a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` that converts MNIST targets 
    to one-hot encoding, trains an HPELM model, evaluates its performance, and returns test accuracy.
    Use L2 regularization to avoid overfitting.
    Return only function code and imports, remove examples or other python code after the function.

    - Input values: X_train and X_test have shape (num_samples, 784), y_train and y_test are 1D arrays of shape (num_samples,)
    - Task is a 10-class classification problem
"""

# # Append the HPELM info to the prompt
# with open('hpelm_doc.md', 'r') as file:
#     full_prompt = prompt + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()

# Append the ELM implementation to the prompt
with open('hpelm_doc_alt.md', 'r') as file:
    full_prompt = prompt.replace("HPELM", "ELM") + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()


# response = py_agent.run_sync(full_prompt)
response = simple_agent.run_sync(full_prompt)
generated_code = response.data

print(generated_code)
print(response.usage())

# %%

# # small ai agent that extracts function code from the response
# class FunctionCodeExtractor(BaseModel):
#     code: str

#     @field_validator('code', mode='after')
#     @classmethod
#     def exec_code(cls, v):
#         print(v)
#         temp_ns = {}
#         try:
#             exec(v, temp_ns)
#         except Exception as e:
#             raise ValueError(f"Error executing code: {e}")
#         return v


# # ollama client
# smol_agent = Agent(ollama_model, result_type=FunctionCodeExtractor, retries=3)


# %% 
# Extract the function code from the response

request = f"Extract and clean the code to be run by exec(code) in Python.: {generated_code}"

exc = None

for _ in range(3):
    if exc is None:
        output = generated_code
    else:
        output = simple_agent.run_sync(request + "\n Last run error: {e}").data

    output_code = output.split("```python")[1].split("```")[0].strip() if "```python" in output else output
    try:
        exec(output_code)
        print("Code executed successfully.\n")
        break
    except Exception as e:
        exc = e
        print(e)

print(output_code)

# %%

# Example usage of the generated function
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
# train test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

accuracy = train_hpelm_mnist(X_train, y_train, X_test, y_test)
print(f"Model accuracy: {accuracy}")


# %%
# fix code in a loop

if False:
    fixed_code = output_code

    for _ in range(3):
        try:
            exec(fixed_code)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            accuracy = train_hpelm_mnist(X_train, y_train, X_test, y_test)
            print(f"Model accuracy: {accuracy}")
            break

        except Exception as e:
            print(e)
            request = f"""
            Fix the code to be run by exec(code) in Python.: {fixed_code}
            Return only the code, no explanation.

            Error: {e}
            """
            output = simple_agent.run_sync(request).data
            fixed_code = output.split("```python")[1].split("```")[0].strip() if "```python" in output else output

            print("Fixed code:\n", fixed_code)
            continue


# %%
