import pandas as pd
import random
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv('./diabetes.csv')
# print(df.head())

columns = df.columns
output_class = columns[-1]

class_0_data = df[df[output_class] == 0]
class_1_data = df[df[output_class] == 1]

num_samples_class_0 = len(class_0_data)
num_samples_class_1 = len(class_1_data)



arranged_df = pd.concat([class_0_data[:], class_1_data[:]])

print(arranged_df.shape)

train_df, test_df = train_test_split(arranged_df,test_size=0.3,random_state=42)

print(train_df.shape, test_df.shape)

train_feature_data = train_df.iloc[:,:-1].values
test_feature_data = test_df.iloc[:,:-1].values

train_feature_data = train_feature_data / 100
test_feature_data = test_feature_data / 100

train_classes = train_df.iloc[:,-1].values
test_classes = test_df.iloc[:,-1].values

# print(feature_data.shape,classes.shape)
# print(type(feature_data[0][0]),type(classes[0]))
learning_rate = 0.2
weights = []
for i in range(len(train_feature_data[0])):
    temp = random.random()
    weights.append(round(temp,2))

threshold = random.random()
threshold = round(threshold,2)
print(f"Random Weights = {weights} and threshold = {threshold}")
all_good = False
row = 0

print(f"train feature lelngth = {len(train_feature_data)}")
iterations = 500
# Train on train dataset
while row<num_samples_class_0 and iterations>0:
    iterations -= 1
    # print(iterations)
    sum_weights = 0
    
    for col in range(len(train_feature_data[0])):
        sum_weights += train_feature_data[row][col]*weights[col]
    if sum_weights>=threshold:
        pred_class = 1
        if pred_class!=train_classes[row]:
            weights -= (learning_rate*train_feature_data[row])
            # print(weights)
            row = 0
            continue
    
    row += 1
    if row==num_samples_class_0:
        print(row)

print(f"First Adapted Weights = {weights}")

iterations = 500
row = num_samples_class_0
while row<len(train_feature_data) and iterations>0:
    iterations -= 1
    # print(iterations)
    sum_weights = 0
    
    for col in range(len(train_feature_data[0])):
        sum_weights += train_feature_data[row][col]*weights[col]

    if sum_weights<threshold:
        pred_class = 0
        if pred_class!=train_classes[row]:
            weights += (learning_rate*train_feature_data[row])
            # print(weights)
            row = num_samples_class_0
            continue
    row += 1
    if row==num_samples_class_1:
        print(row)


print(f"Second Adapted Weights = {weights}")
# Prediction on test dataset
predicted_classes = []
for i in range(len(test_feature_data)):
    sum_weights = 0
    for j in range(len(test_feature_data[0])):
        sum_weights += test_feature_data[i][j]*weights[j]
    if sum_weights>=threshold:
        predicted_classes.append(1)
    else:
        predicted_classes.append(0)


# print(weights)
res = predicted_classes==test_classes

count = 0

for i in range(len(predicted_classes)):
    if predicted_classes[i]==test_classes[i]:
        count+=1

pred_class_count = Counter(predicted_classes)
actual_class_count = Counter(test_classes)

print(f"Accuracy= {count/len(test_classes)}")
# if False in res:
#     print(False)
#     print("There is some problem")
# else:
#     print(True)
#     print("All tests are passed")

if pred_class_count==actual_class_count:
    print("Passed")


# query = list(map(int,input("enter 10 bits sequence: ").split()))[:10]
# if query:
#     total = query*weights

#     if sum(total)>=threshold:
#         print("class 1")
#     else:
#         print("Class 0")
# else:
#     pass