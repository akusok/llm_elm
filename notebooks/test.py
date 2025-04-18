from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import nest_asyncio
nest_asyncio.apply()

# %%

class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIModel(
    # model_name='llama3.1',
    model_name='qwen2.5:latest',
    # model_name='mistral-small3.1',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
agent = Agent(ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""

# %%
# python function model

class PythonFunctionModel(BaseModel):
    body: str

# ollama client
py_agent = Agent(ollama_model, result_type=PythonFunctionModel, retries=3)

# Define a prompt for generating a Python function
prompt = """
Write and return the code for a Python function that calculates the sum of two numbers.
- Function name: calculate_sum
- Parameters: a, b
- Body: return the sum of a and b
"""

# Generate the code
# Use the Agent class to handle the prompt and response
response = py_agent.run_sync(prompt)

# # Assume the response is a string of Python code
generated_code = response.data.body

# %%

# Create a temporary namespace
temp_ns = {}
exec(generated_code, temp_ns)

# Call the function from the temporary namespace
result = temp_ns['calculate_sum'](1, 2)
print(result)  # Output: 3

# %%

class PythonFunctionModel(BaseModel):
    body: str

# ollama client
py_agent = Agent(ollama_model, result_type=PythonFunctionModel, retries=3)


prompt = """
    Write a Python function that trains an HPELM model on the MNIST dataset and evaluates its performance. Use train/test data split inside the function. Return only the code of the function body.
    - Function name: train_hpelm_mnist
    - Parameters: X, y from fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    - Output: accuracy
"""

# Load the content of 'hpelm_info.md' file
with open('hpelm_info.md', 'r') as file:
    hpelm_info = file.read()

# Append the HPELM info to the prompt
full_prompt = prompt + "\n\n" + "Here is additional context about HPELM usage:\n" + hpelm_info
response = py_agent.run_sync(full_prompt)

# Generate the code
# response = py_agent.run_sync(prompt)

# Assume the response is a string of Python code
generated_code = response.data.body

# Print the generated code
print(generated_code)

# %%

from sklearn.datasets import fetch_openml

# Create a temporary namespace
temp_ns = {}
exec(generated_code, temp_ns)

# Example usage of the generated function
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
accuracy = temp_ns['train_hpelm_mnist'](X, y)
print(f"Model accuracy: {accuracy}")


# %%
