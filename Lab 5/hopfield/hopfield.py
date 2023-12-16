import numpy as np


stored_patterns = np.array([[1,-1,1],[1,-1,-1],[-1,-1,1],[1,1,1],[-1,-1,-1]])
no_of_neurons = len(stored_patterns[0])
# print(no_of_neurons)
# Weights Initialization
W = np.zeros([no_of_neurons,no_of_neurons])


# Update weights for storing
for i in range(len(stored_patterns)):
    for row in range(no_of_neurons):
        for col in range(no_of_neurons):
            if row != col:
                W[row][col] += stored_patterns[i][row]*stored_patterns[i][col]

print(f"Weights after storing patterns = {W}")
input_pattern = np.array([1,1,-1])

# Recall Pattern
print(f"Input Pattern = {input_pattern}")
for i in range(10):
    temp = np.matmul(input_pattern,W)
    temp = [1 if num>=0 else -1 for num in temp]
    input_pattern = temp.copy()
    # print(f"Epoch {i} = {temp}")

print(f"Recalled Pattern = {temp}")