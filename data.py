import json

seq_len = 10
with open('training.json', 'r') as file:
    data = json.load(file)

training_data = []
training_output = []

for sample in range(50):
    training_data.append(data[sample:sample+seq_len])
    training_output.append(data[sample+seq_len])

validation_data = []
validation_output = []

for sample in range(51, 80):
    validation_data.append(data[sample:sample+seq_len])
    validation_output.append(data[sample+seq_len])

# print(training_data[-1][0][0][0][2])
# print(training_output[0][0][0][2])
