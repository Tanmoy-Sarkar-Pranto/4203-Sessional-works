import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("diabetes.csv")
co = data.columns
cls = co[-1]

# Separate features (X) and target variable (y)
X = data.drop(cls, axis=1)
y = data[cls]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_tr.to_numpy()
X_test = X_te.to_numpy()
y_train = y_tr.to_numpy()
y_test = y_te.to_numpy()

# Define the architecture of the neural network
input_size = X_train.shape[1]
hidden_size = 16
output_size = 1
learning_rate = 0.01
num_epochs = 10000

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(output_size)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    # Calculate the loss
    loss = ((output - y_train.reshape(-1, 1)) ** 2).mean()

    # Backpropagation
    #grad_output = 2 * (output - y_train.reshape(-1, 1)) * output * (1 - output)
    #error_hidden = d_output.dot(output_layer.T)
    grad_output = loss * output * (1 - output)
    grad_hidden = np.dot(grad_output, weights_hidden_output.T) * hidden_output * (1 - hidden_output)

    # Update weights and biases
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, grad_output)
    #weights_hidden_output -= learning_rate * (loss/hidden_output)
    bias_output -= learning_rate * grad_output.sum(axis=0)
    weights_input_hidden -= learning_rate * np.dot(X_train.T, grad_hidden)
    #weights_input_hidden -= learning_rate * (loss/X_train)
    bias_hidden -= learning_rate * grad_hidden.sum(axis=0)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

# Testing
hidden_input = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
output = sigmoid(output_layer_input)

predictions = (output > 0.5).astype(int)
accuracy = (predictions == y_test.reshape(-1, 1)).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
