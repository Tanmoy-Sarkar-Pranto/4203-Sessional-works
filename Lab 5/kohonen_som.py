import numpy as np
import matplotlib.pyplot as plt
import random as rand

no_of_inputs = 20
no_of_nodes = 10
learning_rate = 0.6

# Random weight initialization, each row will have weights for one node
weights = np.random.random([no_of_nodes, no_of_inputs ])
# print(weights)

# Inputs

inputs = np.random.randint(2,size=(no_of_inputs,no_of_inputs))
# print(inputs)

# Calculate Distances
new_res = [0]*no_of_inputs

for i in range(10):
    for input in range(no_of_inputs):
        distances = []
        for node in range(no_of_nodes):
            distance = np.sum((inputs[input]-weights[node])**2)
            distances.append(distance)
        # print(f"Distances = {distances} for input = {inputs[input]}")
        index_of_min_distance = np.argmin(distances)
        new_res[input] = index_of_min_distance + 1
        # print(f"Min Distance index = {index_of_min_distance}")

        # update weight
        weights[index_of_min_distance] = weights[index_of_min_distance] + (learning_rate*(inputs[input]-weights[index_of_min_distance]))
        # print(f"Updated weights = {weights}")

        # print(f"{res} and {new_res}")
    if learning_rate>0.05:
        learning_rate -= 0.05
print(f"Final Cluster = {new_res}")
# print(f"Final weights: {weights}")

test_input = [1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1]
distances = []
for node in range(no_of_nodes):
    distance = np.sum((test_input-weights[node])**2)
    distances.append(distance)
class_of_test_input = np.argmin(distances)
print(f"Test Class output: {class_of_test_input}")
