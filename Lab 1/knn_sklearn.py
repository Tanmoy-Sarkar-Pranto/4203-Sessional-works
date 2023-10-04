import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


import math
import matplotlib.pyplot as plt
a = []
with open('input.txt', 'r') as file:
    for line in file:
        a.append(list(line.split(' ')))


heights = a[0]
weights = a[1]
classes = a[2]

heights = [float(x) for x in heights]
weights = [float(x) for x in weights]
classes = [str(x) for x in classes]

X = list(zip(heights, weights))

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X, classes)

query_input = list(map(float,input().split()))[:2]

Y = list(zip([query_input[0]],[query_input[1]]))

predicted_class = knn.predict(Y)

predicted_class = ''.join(predicted_class)

print(f"Predicted Class: {predicted_class}")

plt.figure()

colors = {'F': 'blue', 'W': 'red'}

for i in range(len(heights)):
    # print(classes[i])
    plt.scatter(heights[i], weights[i], c=colors[classes[i]], label=classes[i], marker='o')


plt.scatter(query_input[0],query_input[1],c=colors[predicted_class[0]],marker='+')

plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.title('KNN Classification using Sklearn')

plt.show()

heights.append(query_input[0])
weights.append(query_input[1])
classes.append(predicted_class)

heights = [str(x) for x in heights]
weights = [str(x) for x in weights]
classes = [str(x) for x in classes]

# Combine the lists with appropriate separators
data = [' '.join(heights), ' '.join(weights), ' '.join(classes)]

with open('input.txt', 'w') as file:
    file.write('\n'.join(data))
