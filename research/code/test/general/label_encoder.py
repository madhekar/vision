from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import numpy as np

# Original categorical data
data = ['red', 'green', 'blue', 'red', 'green', 'yellow']

# Initialize LabelEncoder
le = LabelEncoder()

# Fit the encoder to the data and transform it
encoded_data = le.fit_transform(data)

print(f"Original data: {data}")
print(f"Encoded data: {encoded_data}")

# Use inverse_transform to convert encoded data back to original
decoded_data = le.inverse_transform(encoded_data)

print(f"Decoded data: {decoded_data}")

decoded_label = le.inverse_transform([1])

print(literal_eval(str(decoded_label)))

print("".join(decoded_label))
# Example with numerical labels
# numerical_data = [1, 2, 2, 6]
# le_num = LabelEncoder()
# encoded_numerical_data = le_num.fit_transform(numerical_data)
# print(f"\nOriginal numerical data: {numerical_data}")
# print(f"Encoded numerical data: {encoded_numerical_data}")
# decoded_numerical_data = le_num.inverse_transform(encoded_numerical_data)
# print(f"Decoded numerical data: {decoded_numerical_data}")