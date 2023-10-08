import random
import pandas as pd

def generate_binary_numbers(n_bits):
  """Generates all binary numbers of n bits."""
  binary_numbers = []
  for i in range(2**n_bits):
    binary_number = bin(i)[2:].zfill(n_bits)
    binary_numbers.append(binary_number)
  return binary_numbers

def assign_classes(binary_numbers):
  """Assigns class 0 to the first half of the binary numbers and class 1 to the second half."""
  class_labels = []
  for i in range(len(binary_numbers) // 2):
    class_labels.append(0)
  for i in range(len(binary_numbers) // 2, len(binary_numbers)):
    class_labels.append(1)
  return class_labels

# Generate 2^10 binary numbers of 10 bits
bit_size = 10
binary_numbers = generate_binary_numbers(bit_size)

# print(binary_numbers)
# Extract the individual bits into a new list
bits = []
for binary_number in binary_numbers:
  bits.append(list(binary_number))

# print(bits)
class_labels = assign_classes(binary_numbers)
# print(class_labels)
# Create a pandas DataFrame


df = pd.DataFrame({
 
    "bit_0": [i[0] for i in bits],
    "bit_1": [i[1] for i in bits],
    "bit_2": [i[2] for i in bits],
    "bit_3": [i[3] for i in bits],
    "bit_4": [i[4] for i in bits],
    "bit_5": [i[5] for i in bits],
    "bit_6": [i[6] for i in bits],
    "bit_7": [i[7] for i in bits],
    "bit_8": [i[8] for i in bits],
    "bit_9": [i[9] for i in bits],
    "class": class_labels,

})

# Save the DataFrame to an Excel file
# df.to_excel("binary_numbers.xlsx", index=False)

df.to_csv(f'new_data_{bit_size}_bits.csv', index = False, encoding='utf-8') # False: not include index
# print(df)