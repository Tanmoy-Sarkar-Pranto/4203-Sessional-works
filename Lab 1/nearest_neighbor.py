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
# query_input = [1.69, 79.5]

query_input = list(map(float,input().split()))[:2]

nearest_class = None
nearest_distance = float('inf')

def calc_distance(q,h,w):
    return math.sqrt((q[0]-h)**2+(q[1]-w)**2)

for i in range(len(heights)):
    tem = calc_distance(query_input,heights[i],weights[i])
    if tem<nearest_distance:
        nearest_distance = tem
        nearest_class = classes[i]

print(f"Nearest Neighbor Class: {nearest_class}" )
print(f"Nearest Distance: {nearest_distance}")

class_colors = {'F':'blue','W':'red'}

for i in range(len(heights)):
    # print(classes[i])
    plt.scatter(heights[i], weights[i], c=class_colors[classes[i]], label=classes[i], marker='o')
plt.scatter(query_input[0],query_input[1],c=class_colors[nearest_class],label=nearest_class,marker='+')

plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.title('Nearest Neighbor Classification')

plt.show()

heights.append(query_input[0])
weights.append(query_input[1])
classes.append(nearest_class)

heights = [str(x) for x in heights]
weights = [str(x) for x in weights]
classes = [str(x) for x in classes]

# Combine the lists with appropriate separators
data = [' '.join(heights), ' '.join(weights), ' '.join(classes)]

with open('input.txt', 'w') as file:
    file.write('\n'.join(data))