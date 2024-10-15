import json
import numpy as np
import matplotlib.pyplot as plt

seq_len = 1
with open('training.json', 'r') as file:
    data = json.load(file)
    data = np.array(data)

training_data = []
training_output = []

for sample in range(8000):
    training_data.append(data[sample:sample+seq_len])
    training_output.append(data[sample+seq_len])

validation_data = []
validation_output = []

for sample in range(8000, 8500):
    validation_data.append(data[sample:sample+seq_len])
    validation_output.append(data[sample+seq_len])

example = data[8500:8501]
example_output = data[8501]

def U_Array(data_point):
    return data_point[:, :, 0]

def V_Array(data_point):
    return data_point[:, :, 1]

def W_Array(data_point):
    return data_point[:, :, 2]

def Print_U_V_W(data_point, data_point_NN):
    x = np.array([i for i in range(64)])
    z = x
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    imU = axes[0, 0].imshow(U_Array(data_point), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[0, 0].set_title('Heatmap of U(x, z)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('z')
    fig.colorbar(imU, ax=axes[0, 0], label='U(x, z)')

    imV = axes[0, 1].imshow(V_Array(data_point), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[0, 1].set_title('Heatmap of V(x, z)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('z')
    fig.colorbar(imV, ax=axes[0, 1], label='V(x, z)')

    imW = axes[0, 2].imshow(W_Array(data_point), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[0, 2].set_title('Heatmap of W(x, z)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('z')
    fig.colorbar(imW, ax=axes[0, 2], label='W(x, z)')

    imU_NN = axes[1, 0].imshow(U_Array(data_point_NN), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[1, 0].set_title('Heatmap of U(x, z) (Predicted)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z')
    fig.colorbar(imU_NN, ax=axes[1, 0], label='U(x, z)')

    imV_NN = axes[1, 1].imshow(V_Array(data_point_NN), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[1, 1].set_title('Heatmap of V(x, z) (Predicted)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('z')
    fig.colorbar(imV_NN, ax=axes[1, 1], label='V(x, z)')

    imW_NN = axes[1, 2].imshow(W_Array(data_point_NN), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[1, 2].set_title('Heatmap of W(x, z) (Predicted)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('z')
    fig.colorbar(imW_NN, ax=axes[1, 2], label='W(x, z)')

    plt.tight_layout()
    plt.show()

