from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from hpelm import ELM
import numpy as np

def train_hpelm_mnist(X_train, y_train, X_test, y_test):
    # Ensure targets are one-hot encoded for classification task
    num_classes = 10
    
    # Create and configure the HPELM model for classification with sigmoid neurons
    elm = ELM(X_train.shape[1], num_classes, classification="c")
    
    # Add 50 sigmoid neurons to the model
    elm.add_neurons(50, "sigm")
    
    # Train the model using training data
    elm.train(X=X_train, T=y_train)
    
    # Make predictions on test data
    y_pred = elm.predict(X_test)
    
    # Calculate accuracy by comparing predicted labels with true labels
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test)
    
    return accuracy


# Example usage of the generated function
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

y = np.eye(10)[y.astype(int)]
# train test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print size of X and y arrays
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

accuracy = train_hpelm_mnist(X_train, y_train, X_test, y_test)

print(f"Model accuracy: {accuracy}")
