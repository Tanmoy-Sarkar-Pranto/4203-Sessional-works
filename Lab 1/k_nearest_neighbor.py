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
k=5
print(classes)
query_input = list(map(float,input().split()))[:2]
# query_input = [1.69, 79.5]
def calc_distance(q,h,w):
    return math.sqrt((q[0]-h)**2+(q[1]-w)**2)

distances = []

for i in range(len(heights)):
    tem = calc_distance(query_input,heights[i],weights[i])
    distances.append((tem,classes[i]))

distances.sort()
print(distances)
class_count = {}

for i in range(k):
    distance, cls = distances[i]
    if cls in class_count:
        class_count[cls] += 1
    else:
        class_count[cls] = 1

final_class = max(class_count, key=class_count.get)
# print(final_class)
# print(distances)
print(f"Nearest Neighbor Class: {final_class}" )

class_colors = {'W': 'red', 'F': 'blue'}

for i in range(len(heights)):
    # print(classes[i])
    plt.scatter(heights[i], weights[i], c=class_colors[classes[i]], label=classes[i], marker='o')
plt.scatter(query_input[0],query_input[1],c=class_colors[final_class],label=final_class,marker='+')
plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.title('KNN Classification')

plt.show()

heights.append(query_input[0])
weights.append(query_input[1])
classes.append(final_class)

heights = [str(x) for x in heights]
weights = [str(x) for x in weights]
classes = [str(x) for x in classes]

# Combine the lists with appropriate separators
data = [' '.join(heights), ' '.join(weights), ' '.join(classes)]

with open('input.txt', 'w') as file:
    file.write('\n'.join(data))
