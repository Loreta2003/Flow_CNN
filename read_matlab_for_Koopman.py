import h5py
import json
import numpy as np

training_dataset = []
validation_dataset = []
with h5py.File('../minimal_2D_UVW_yplus_20.mat', 'r') as file:  
    U_data = file['U_2D'][:]
    #V_data = file['V_2D'][:]
    #W_data = file['W_2D'][:]
    for seq_len in range(9000):
        if seq_len < 8000:
            training_dataset.append([])
            for x_coord in range(64):
                training_dataset[seq_len].append([])
                for z_coord in range(64):
                    training_dataset[seq_len][x_coord].append([U_data[x_coord][z_coord][seq_len]])
        if seq_len >= 8000:
            validation_dataset.append([])
            for x_coord in range(64):
                validation_dataset[seq_len-8000].append([])
                for z_coord in range(64):
                    validation_dataset[seq_len-8000][x_coord].append([U_data[x_coord][z_coord][seq_len]])
    with open('training.json', 'w') as file:
        json.dump(training_dataset, file)
    with open('validation.json', 'w') as file:
        json.dump(validation_dataset, file)

training_dataset = np.array(training_dataset)
print(training_dataset.shape)
validation_dataset = np.array(validation_dataset)
print(validation_dataset.shape)
