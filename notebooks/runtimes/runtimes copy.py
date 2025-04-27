
# %%

import time
import json
import requests
import pandas as pd

import nest_asyncio
nest_asyncio.apply()


# %%
# setup ollama

model_names=[
    "llama3.2",
    'qwen2.5-coder:7b',
    "llama3.1",
    # "phi4",
    # "marco-o1",
    # "qwen2.5-coder:32b",
    # "deepseek-r1:8b",
]

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
with open('../../doc/hpelm_doc_alt.md', 'r') as file:
    full_prompt = prompt + "\n\n" + "Here is additional context about HPELM usage:\n" + file.read()

fix_request = """
Fix the code to be run by exec(code) in Python.: {generated_code}
Return only the code, no explanation.

Error: {msg}
"""

# %%


def stream_ollama(prompt, model="llama3.2", base_url="http://localhost:11434"):
    url = f"{base_url}/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    t = time.time()
    t1 = None
    n = 0
    with requests.post(url, json=data, headers=headers, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                try:
                    # Ollama streams JSON lines, each with a 'message' key
                    msg = json.loads(line)
                    token = msg.get("message", {}).get("content", "")
                    # print(token, end="", flush=True)
                    n += 1
                    if t1 is None:
                        t1 = time.time() - t
                except Exception as e:
                    print(f"\n[Error parsing line: {e}]\n{line}")

    return time.time()-t, t1, n

x = stream_ollama("hello", model="llama3.2", base_url="http://localhost:11434")
print()
print(x)


# %%

import time

results = []

for model_name in model_names:
    print(f"\nRunning model: {model_name}")

    for j in range(11):
        print(j)
        t, t1, n = stream_ollama(full_prompt, model=model_name, base_url="http://localhost:11434")
        results.append({
            "j": j,
            "model_name": model_name,
            "runtime_sec": t,
            "first_token_sec": t1,
            "num_tokens": n,
        })

        # Save results
        pd.DataFrame(results).to_pickle("runtimes_results_2.pkl")

# %%
