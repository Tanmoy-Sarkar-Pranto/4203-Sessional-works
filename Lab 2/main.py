import pandas as pd
import random

df = pd.read_csv('./binary_numbers_csv.csv')
# print(df.head())

columns = df.columns
feature_data = df.iloc[:,:-1].values

classes = df.iloc[:,-1].values

# print(feature_data.shape,classes.shape)
# print(type(feature_data[0][0]),type(classes[0]))

weights = []
for i in range(len(feature_data[0])):
    temp = random.random()
    weights.append(round(temp,2))

threshold = random.random()
threshold = round(threshold,2)
print(weights,threshold)
all_good = False
row = 0

print(len(feature_data))
while row<len(feature_data):
    sum_weights = 0
    
    for col in range(len(feature_data[0])):
        sum_weights += feature_data[row][col]*weights[col]
    if sum_weights>=threshold:
        pred_class = 1
        if pred_class!=classes[row]:
            weights -= feature_data[row]
            # print(weights)
            row = 0
            continue
    else:
        pred_class = 0
        if pred_class!=classes[row]:
            weights += feature_data[row]
            # print(weights)
            row = 0
            continue
    row += 1
    if row==len(feature_data):
        print(row)

predicted_classes = []
for i in range(len(feature_data)):
    sum_weights = 0
    for j in range(len(feature_data[0])):
        sum_weights += feature_data[i][j]*weights[j]
    if sum_weights>=threshold:
        predicted_classes.append(1)
    else:
        predicted_classes.append(0)


print(weights)
res = predicted_classes==classes
if False in res:
    print(False)
else:
    print(True)