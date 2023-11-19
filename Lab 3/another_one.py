import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_size = 8
hidden_size = 20
output_size = 1

# np.random.seed(0)

input_layer = np.random.random((input_size, hidden_size))
output_layer = np.random.random((hidden_size, output_size))

learning_rate = 0.01
epochs = 10000

# Read data from a CSV file
data = pd.read_csv('./diabetes.csv')  # Replace 'your_input_file.csv' with the actual CSV file path
X = data.drop('Outcome', axis=1).values  # Assuming 'output' is the target column
y = data['Outcome'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

for epoch in range(epochs):

    # Forward Pass
    hidden_input = np.dot(X_train, input_layer) 
    hidden_output = sigmoid(hidden_input) 
    output_input = np.dot(hidden_output, output_layer) 
    output = sigmoid(output_input) 
    
    error = ((output-y_train.reshape(-1,1))**2).mean()
    # if epoch%100==0:
    #     print(f"Error:{np.mean(error)}")
    # backward pass
    d_output = error * sigmoid_derivative(output) 
    
    error_hidden = d_output.dot(output_layer.T) 

    d_hidden = error_hidden * sigmoid_derivative(hidden_output) 
    
    output_layer -= hidden_output.T.dot(d_output) * learning_rate 
    input_layer -= X_train.T.dot(d_hidden) * learning_rate 
    if epoch%1000==0:
        plt.plot(epoch,error)
# print(error)
# Now you can use the trained model for prediction as before
# test_X = np.array([[1, 1, 0], [1, 1, 1]])
hidden_input = np.dot(X_test, input_layer)
hidden_output = sigmoid(hidden_input)

output_input = np.dot(hidden_output, output_layer)
predicted_output = sigmoid(output_input)

# print("Predicted Output:")
for i in range(len(predicted_output)):
    predicted_output[i] = predicted_output[i].round()

result = predicted_output == y_test
# print()
# print(result)
count = 0
# print(predicted_output)
accuracy = (predicted_output==y_test.reshape(-1,1)).mean()
print(f"Test accuracy:{accuracy*100}%")
