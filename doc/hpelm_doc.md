# HPELM Quick Reference Guide

## Installation

```bash
pip install hpelm
```

## Basic Usage

### Import

```python
from hpelm import HPELM
import numpy as np
```

### Create a HPELM Model

```python
# For regression: inputs=10, outputs=1
elm = HPELM(10, 1)

# For classification: inputs=10, outputs=3 (3 classes)
elm = HPELM(10, 3, classification="c")

# For weighted classification
elm = HPELM(10, 3, classification="wc", w=[1.0, 2.0, 1.5])

# For multi-label classification
elm = HPELM(10, 3, classification="ml")
```

### Adding Neurons

```python
# Add 20 sigmoid neurons
elm.add_neurons(20, "sigm")

# Add 10 linear neurons
elm.add_neurons(10, "lin")

# Add 15 tanh neurons
elm.add_neurons(15, "tanh")

# Add RBF neurons with different distance metrics
elm.add_neurons(10, "rbf_l1")    # L1 distance
elm.add_neurons(10, "rbf_l2")    # L2 distance
elm.add_neurons(10, "rbf_linf")  # L∞ distance
```

### Training with NumPy Arrays

```python
# Prepare data
X_train = np.random.rand(1000, 10)  # 1000 samples, 10 features
T_train = np.random.rand(1000, 1)   # For regression

# For classification, one-hot encode targets
T_class = np.zeros((1000, 3))
for i in range(1000):
    T_class[i, np.random.randint(0, 3)] = 1

# Train the model
elm.train(X_train, T_train)  # Regression
elm.train(X_train, T_class, "c")  # Classification
```

### Making Predictions

```python
# With NumPy arrays
X_test = np.random.rand(100, 10)
Y_pred = elm.predict(X_test)

# With HDF5 files
elm.predict("X_test.h5", "Y_pred.h5")

# With specific range
elm.predict("X_test.h5", "Y_pred.h5", istart=0, icount=50)
```

### Calculating Error

```python
# Calculate error with NumPy arrays
error = elm.error(T_test, Y_pred)

# Calculate error with HDF5 files
error = elm.error("T_test.h5", "Y_pred.h5")

# Calculate error on a specific range
error = elm.error("T_test.h5", "Y_pred.h5", istart=0, icount=50)
```

## Tips and Best Practices

- Set appropriate batch size according to available memory
- For classification tasks, ensure outputs are one-hot encoded
- Use validation_corr() to find optimal number of neurons
- L2 regularization can be controlled via the `norm` parameter
