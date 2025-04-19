# LLM-ELM

An LLM expert system for writing ML solution based on Extreme Learning Machines

## text-query

`text-query`: Using text-only queries, no Pydantic structured output. Small models struggle a lot with the structured output, while producing good results with bare-text output.

### Experiment: LLM-assisted ELM Code Generation and Evaluation

The [`notebooks/text_query.py`](notebooks/text_query.py) notebook demonstrates an experiment where an LLM agent is used to generate Python code for training an Extreme Learning Machine (ELM) on the MNIST dataset. The workflow is as follows:

1. **Agent Setup:**  
   An LLM agent is configured using the `pydantic_ai` library with an Ollama-compatible OpenAI API.

2. **Structured Output Example:**  
   The agent is prompted for structured answers (e.g., extracting city and country from a question).

3. **Code Generation:**  
   The agent is prompted to generate a Python function `train_hpelm_mnist(X_train, y_train, X_test, y_test)` that:
   - Converts MNIST targets to one-hot encoding
   - Trains an ELM model (using the `hpelm` library)
   - Evaluates test accuracy
   - Uses L2 regularization to avoid overfitting

4. **Code Extraction and Execution:**  
   The generated code is extracted, cleaned, and executed dynamically.

5. **Evaluation:**  
   The generated function is run on the MNIST dataset (fetched via `sklearn.datasets.fetch_openml`), and the resulting model accuracy is printed.

This experiment showcases how LLMs can automate the creation and evaluation of machine learning pipelines using domain-specific libraries like `hpelm`.


