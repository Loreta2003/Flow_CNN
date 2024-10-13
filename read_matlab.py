import h5py
import json

dataset = []
with h5py.File('minimal_2D_UVW_yplus_20.mat', 'r') as file:  
    U_data = file['U_2D'][:]
    V_data = file['V_2D'][:]
    W_data = file['W_2D'][:]
    for seq_len in range(100):
        dataset.append([])
        for x_coord in range(64):
            dataset[seq_len].append([])
            for z_coord in range(64):
                dataset[seq_len][x_coord].append([U_data[x_coord][z_coord][seq_len], V_data[x_coord][z_coord][seq_len], W_data[x_coord][z_coord][seq_len]])
    with open('training.json', 'w') as file:
        json.dump(dataset, file)
