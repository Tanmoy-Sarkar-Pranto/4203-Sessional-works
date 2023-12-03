import numpy as np
import matplotlib.pyplot as plt

epoch_losses = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W1, b1, W2, b2):
    # Calculate the output of the hidden layer
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    # Calculate the output of the output layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    return a1, a2

def train(X, Y, W1, b1, W2, b2, epochs=3000, learning_rate=0.1):
    for epoch in range(epochs):
        # Forward propagation
        A1, A2 = forward(X, W1, b1, W2, b2)

        # Calculate the error
        E = Y - A2

        # Calculate the delta for the output layer
        delta2 = E * (A2 * (1 - A2))

        # Calculate the delta for the hidden layer
        delta1 = np.dot(delta2, W2.T) * (A1 * (1 - A1))

        # Update the weights and biases
        W2 += learning_rate * np.dot(A1.T, delta2)
        b2 += learning_rate * np.sum(delta2, axis=0)
        W1 += learning_rate * np.dot(X.T, delta1)
        b1 += learning_rate * np.sum(delta1, axis=0)

        # Calculate the loss
        loss = np.mean(np.power(E, 2))

        if epoch % 300 == 0:
            print(f"Epoch {epoch}: loss = {loss}")
        epoch_losses.append(loss)

# Prepare the training data
X = np.array([[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1],
    [1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]
])
Y = np.array([
    [0],[1],[1],[0],
    [1],[0],[0],[1]
])

# Initialize weights and biases randomly
W1 = np.random.randn(X.shape[1], 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 1)
b2 = np.random.randn(1)

# Train the model
train(X, Y, W1, b1, W2, b2)

# Make predictions
_, predictions = forward(X, W1, b1, W2, b2)
# print(predictions)
for i in range(len(predictions)):
    predictions[i] = predictions[i].round()
    
def calculate_accuracy(predictions, labels):
    correct_predictions = np.sum(predictions == labels)
    total_samples = len(labels)
    accuracy = correct_predictions / total_samples
    return accuracy


def plot_loss_curve(loss_values):
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss Curve')
    plt.legend()
    plt.show()
    
# Calculate and print accuracy
accuracy = calculate_accuracy(predictions, Y)
print(f"Accuracy: {accuracy}")
plot_loss_curve(epoch_losses)
