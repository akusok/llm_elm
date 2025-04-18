import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from hpelm import ELM

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
y_onehot = enc.fit_transform(y.reshape(-1, 1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Train ELM classifier
elm = ELM(X_train.shape[1], y_train.shape[1], classification="c")
elm.add_neurons(1000, "sigm")  # 1000 sigmoid neurons
elm.train(X_train, y_train, "c")

# Predict and evaluate
y_pred = elm.predict(X_test)
accuracy = (y_pred.argmax(axis=1) == y_test.argmax(axis=1)).mean()
print(f"Test accuracy: {accuracy:.4f}")
