# HPELM Quick Guide

## Basic ELM Usage

HPELM (High Performance Extreme Learning Machine) is a Python library for efficient neural network training with fast non-iterative training.

### Installation

```bash
pip install hpelm
```

### Basic Regression

```python
import numpy as np
from hpelm import ELM

# Create ELM with 5 input features and 1 output
elm = ELM(inputs=5, outputs=1)

# Add hidden neurons (20 sigmoid neurons)
elm.add_neurons(20, "sigm")

# Train ELM (X_train is input data, y_train is target values)
elm.train(X_train, y_train)

# Predict on new data
y_pred = elm.predict(X_test)

# Calculate MSE
error = elm.error(y_test, y_pred)
```

### Basic Classification

```python
import numpy as np
from hpelm import ELM

# One-hot encode classes first (for 3 classes)
# Example: class 0 -> [1,0,0], class 1 -> [0,1,0], class 2 -> [0,0,1]
T_onehot = np.eye(3)[y_train.astype(int)]

# Create ELM with 5 input features and 3 output classes
elm = ELM(inputs=5, outputs=3)

# Add hidden neurons
elm.add_neurons(30, "sigm")

# Train ELM for classification
elm.train(X_train, T_onehot, "c")

# Predict classes
y_pred = elm.predict(X_test)
predicted_classes = np.argmax(y_pred, axis=1)

# Calculate classification error
error = elm.error(T_test, y_pred)
```

### Using L2 Regularization

L2 regularization prevents overfitting by penalizing large weights. Use it when:
- Your model shows signs of overfitting
- You have limited training data
- You want to improve generalization

```python
# Create ELM with L2 regularization (norm=0.1)
elm = ELM(inputs=5, outputs=1, norm=0.1)
elm.add_neurons(50, "sigm")
elm.train(X_train, y_train)
```

### Different Neuron Types

```python
# Sigmoid neurons (default)
elm.add_neurons(20, "sigm")

# Hyperbolic tangent
elm.add_neurons(20, "tanh")

# Linear neurons
elm.add_neurons(10, "lin")

# Radial basis functions
elm.add_neurons(30, "rbf_l2")
```

### Batch Processing for Large Datasets

```python
# Use batch size of 500 samples
elm = ELM(inputs=5, outputs=1, batch=500)
elm.add_neurons(30, "sigm")
elm.train(X_train, y_train)
