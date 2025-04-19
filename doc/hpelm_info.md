# HPELM Library Overview

HPELM (High-Performance Extreme Learning Machine) is a Python library that implements the Extreme Learning Machine (ELM) algorithm with various optimizations for speed and efficiency. The library provides two main classes: `ELM` for standard implementation and HPELM for high-performance implementation designed to work with large datasets.

## Core Concepts

ELMs are feed-forward neural networks with a single hidden layer where the weights between the input and hidden layer are randomly initialized and never updated. Only the weights between the hidden layer and output are trained (using a simple least-squares solution). This makes ELM training extremely fast compared to backpropagation methods.

## Main Classes

### 1. ELM Class

The basic implementation of Extreme Learning Machine:

```python
from hpelm import ELM

# Create an ELM with 10 input features and 1 output
elm = ELM(10, 1)
```

### 2. HPELM Class

High-performance implementation for larger datasets:

```python
from hpelm import HPELM

# Create a HPELM with 10 input features and 1 output
hpelm = HPELM(10, 1)
```

## Key Features

### Initialization Parameters

Both `ELM` and HPELM classes accept these common parameters:

- `inputs`: Number of input features
- `outputs`: Number of outputs (or classes for classification)
- `classification`: Type of classification task ('c' for standard, 'wc' for weighted, 'ml' for multi-label)
- `w`: Class weights for weighted classification
- `batch`: Batch size for processing data (default 1000)
- `accelerator`: Acceleration type ('GPU' for GPU acceleration, 'basic' for slower solver)
- `precision`: Data precision ('single'/'32' or 'double'/'64')
- `norm`: L2-normalization parameter
- `tprint`: Progress reporting interval in seconds

### Adding Neurons◊

Before training, neurons must be added to the hidden layer:

```python
# Add 20 sigmoidal neurons
elm.add_neurons(20, "sigm")

# Add 10 linear neurons
elm.add_neurons(10, "lin")

# Add 15 hyperbolic tangent neurons
elm.add_neurons(15, "tanh")

# Add 30 radial basis function neurons with L2 norm
elm.add_neurons(30, "rbf_l2")
```

Supported neuron types:
- `lin`: Linear neurons
- `sigm`: Sigmoid neurons
- `tanh`: Hyperbolic tangent neurons
- `rbf_l1`, `rbf_l2`, `rbf_linf`: Radial basis function neurons with different norms

### Training Models

Basic training:

```python
# Simple training
elm.train(X, T)

# For classification
elm.train(X, T, 'c')  # standard classification
elm.train(X, T, 'wc', w=[1, 2])  # weighted classification
elm.train(X, T, 'ml')  # multi-label classification
```

Model structure selection methods:

```python
# Validation set approach
elm.train(X, T, 'V', Xv=X_validation, Tv=T_validation)

# Cross-validation approach
elm.train(X, T, 'CV', k=5)  # 5-fold cross-validation

# Leave-One-Out validation
elm.train(X, T, 'LOO')
```

Optimal pruning (L1 regularization):

```python
# Validation with optimal pruning
elm.train(X, T, 'V', 'OP', Xv=X_validation, Tv=T_validation)

# Cross-validation with optimal pruning
elm.train(X, T, 'CV', 'OP', k=5)

# Leave-One-Out with optimal pruning
elm.train(X, T, 'LOO', 'OP')
```

### Making Predictions

```python
# Get predictions
Y = elm.predict(X_test)
```

### Evaluating Performance

```python
# Calculate error between true values and predictions
error = elm.error(T_test, Y)

# Generate confusion matrix for classification tasks
conf_matrix = elm.confusion(T_test, Y)
```

### Saving and Loading Models

```python
# Save the model
elm.save("model.elm")

# Load the model
new_elm = ELM(10, 1)
new_elm.load("model.elm")

# Alternative serialization to bytes
model_bytes = elm.dumps()
elm.loads(model_bytes)
```

### Additional Features

#### Accessing Hidden Layer Representation

```python
# Get the hidden layer representation of input data
H = elm.project(X)
```

## High-Performance Features (HPELM)

HPELM includes additional functionality for handling larger datasets:

### Validation with Correlation Matrices

```python
# Perform validation using pre-computed correlation matrices
hpelm.validation_corr(fHH, fHT, X_validation, T_validation, steps=10)
```

### GPU Acceleration

HPELM supports GPU acceleration through scikit-CUDA:

```python
# Create GPU-accelerated ELM
elm_gpu = ELM(10, 1, accelerator="GPU")
```

## Examples

### Classification Example

```python
from hpelm import ELM
import numpy as np

# Create synthetic data
X = np.random.randn(1000, 10)
T = np.zeros((1000, 3))
T[np.arange(1000), np.random.randint(0, 3, 1000)] = 1  # One-hot encoding

# Create and configure ELM
elm = ELM(10, 3)  # 10 inputs, 3 classes
elm.add_neurons(20, "sigm")
elm.add_neurons(10, "tanh")

# Train for classification
elm.train(X, T, "c")

# Make predictions
Y = elm.predict(X)
accuracy = 1.0 - elm.error(T, Y)
print(f"Accuracy: {accuracy*100:.2f}%")
```

### Regression Example

```python
from hpelm import ELM
import numpy as np

# Create synthetic data
X = np.random.randn(1000, 5)
T = np.sin(X.sum(axis=1)).reshape(-1, 1)

# Create and configure ELM
elm = ELM(5, 1)  # 5 inputs, 1 output
elm.add_neurons(30, "sigm")
elm.add_neurons(30, "lin")

# Train for regression
elm.train(X, T)

# Make predictions and calculate MSE
Y = elm.predict(X)
mse = elm.error(T, Y)
print(f"Mean Squared Error: {mse:.6f}")
```

## Best Practices

1. **Neuron Selection**: Using a mix of neuron types (linear, sigmoid, tanh, RBF) often improves performance.
2. **Validation**: Use model structure selection methods like validation or cross-validation to find the optimal number of neurons.
3. **Optimal Pruning**: For smaller models with better generalization, use optimal pruning ('OP').
4. **Batch Size**: Adjust the batch size according to your memory constraints.
5. **Precision**: Use double precision for accurate results or single precision for faster computations, especially with GPU.

## Limitations

1. ELMs may require more hidden neurons than other neural networks.
2. Model structure selection methods require all data to fit in memory.
3. Training becomes slower after ~3000 neurons due to computational complexity.

This library is particularly useful for problems requiring fast training and reasonable accuracy when the exact optimal solution isn't required.