# %%
from pydantic import BaseModel, field_validator

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import nest_asyncio
nest_asyncio.apply()

# %%

class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIModel(
    model_name='llama3.1',
    # model_name='qwen2.5:latest',
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

class PythonFunctionModel(BaseModel):
    body: str

# ollama client
py_agent = Agent(ollama_model)


prompt = """
    Write a Python function that trains an HPELM model on the MNIST dataset and evaluates its performance. Add train/test data split inside the function. Return only function code.
    - Function name: train_hpelm_mnist
    - Parameters: X_train, y_train, X_test, y_test
    - Output: accuracy
"""

# Load the content of 'hpelm_info.md' file
with open('hpelm_doc.md', 'r') as file:
    hpelm_info = file.read()

# Append the HPELM info to the prompt
full_prompt = prompt + "\n\n" + "Here is additional context about HPELM usage:\n" + hpelm_info
response = py_agent.run_sync(full_prompt)

# Generate the code
# response = py_agent.run_sync(prompt)

# Assume the response is a string of Python code
generated_code = response.data

print(generated_code)

# %%

# small ai agent that extracts function code from the response
class FunctionCodeExtractor(BaseModel):
    code: str

    @field_validator('code', mode='before')
    @classmethod
    def exec_code(cls, v):
        temp_ns = {}
        exec(v, temp_ns)
        return v


# ollama client
smol_agent = Agent(ollama_model, result_type=FunctionCodeExtractor, retries=3)

# Extract the function code from the response
response = smol_agent.run_sync(
    f"""
    Extract and clean the code to be run by exec(code) in Python.:
    {generated_code}
    """
)

# print response
code = response.data.code
print(code)


# %%

# Example usage of the generated function
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
# train test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Create a temporary namespace
temp_ns = {}
exec(code, temp_ns)

accuracy = temp_ns['train_hpelm_mnist'](X_train, y_train, X_test, y_test)
print(f"Model accuracy: {accuracy}")


# %%

