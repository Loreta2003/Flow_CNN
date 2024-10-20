from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers, models
import model as md
import numpy as np
import data as dt
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import interp1d

cnn_predictor = load_model('flow_CNN.h5')

with open('training.json', 'r') as file:
    data = json.load(file)
    data = np.array(data)

mean_DNS = np.mean(data)
variance_DNS = np.var(data)

def Pointwise_Loss(true_output, predicted_output, loss_function):
    if loss_function == "mse":
        loss = np.square(true_output - predicted_output)
    elif loss_function == "mae":
        loss = np.abs(true_output - predicted_output)
    else:
        raise ValueError("Unknown loss function. Choose 'mse' or 'mae'.")
    return loss

def Average_Loss(true_output, predicted_output, loss_function):
    return np.mean(Pointwise_Loss(true_output, predicted_output, loss_function))

def Loss_Of_Frames(data, loss_function):
    loss_array = []
    for i in range(len(data) - 1):
        frame = data[i]
        true_output = data[i+1]
        predicted_output = Predict_After_Time_T(frame, 1)
        loss_array.append(Average_Loss(true_output, predicted_output, loss_function))
    np.save(f'Frame_Loss_{loss_function}.npy', loss_array)
    return np.array(loss_array)

def Loss_Of_Dataset(data, loss_function):
    return np.mean(np.load(f'Frame_Loss_{loss_function}'))

def Normalized_MSE_Loss_Of_Dataset(data):
    return Loss_Of_Dataset(data, "mse")/variance_DNS

def MSE_Loss_Over_Time(start_time, time_length):
    loss_array = []
    predicted_frames = Predict_During_Time_T(data[start_time], time_length)
    true_frames = data[start_time: start_time+time_length + 1]
    for i in range(len(predicted_frames)):
        loss_array.append(Average_Loss(true_frames[i], predicted_frames[i], "mse"))
    
    loss_array = np.array(loss_array)
    indices = np.arange(len(true_frames))
    interpolation_function = interp1d(indices, loss_array, kind='linear')
    new_indices = np.linspace(0, len(true_frames) - 1, num=500)
    interpolated_loss_array = interpolation_function(new_indices)
    plt.plot(new_indices, interpolated_loss_array, color="red")

    plt.title("Loss vs Time")
    plt.xlabel("Time")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.xticks(indices)
    plt.legend()
    plt.show()

def Normalized_MSE_Loss_Over_Time(start_time, time_length):
    loss_array = []
    predicted_frames = Predict_During_Time_T(data[start_time], time_length)
    true_frames = data[start_time: start_time+time_length + 1]
    var_DNS = np.var(true_frames)
    for i in range(len(predicted_frames)):
        loss_array.append(Average_Loss(true_frames[i], predicted_frames[i], "mse")/var_DNS)
    
    loss_array = np.array(loss_array)
    indices = np.arange(len(true_frames))
    interpolation_function = interp1d(indices, loss_array, kind='linear')
    new_indices = np.linspace(0, len(true_frames) - 1, num=500)
    interpolated_loss_array = interpolation_function(new_indices)
    plt.plot(new_indices, interpolated_loss_array, color="red")

    plt.title("Loss vs Time")
    plt.xlabel("Time")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.xticks(indices)
    plt.legend()
    plt.show()

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

def Print_U_V_W_Animation(data_points, data_points_NN, num_frames):
    x = np.array([i for i in range(64)])
    z = x

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    imU = axes[0, 0].imshow(U_Array(data_points[0]), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[0, 0].set_title('Heatmap of U(x, z)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('z')
    fig.colorbar(imU, ax=axes[0, 0], label='U(x, z)')

    imV = axes[0, 1].imshow(V_Array(data_points[0]), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[0, 1].set_title('Heatmap of V(x, z)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('z')
    fig.colorbar(imV, ax=axes[0, 1], label='V(x, z)')

    imW = axes[0, 2].imshow(W_Array(data_points[0]), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[0, 2].set_title('Heatmap of W(x, z)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('z')
    fig.colorbar(imW, ax=axes[0, 2], label='W(x, z)')

    imU_NN = axes[1, 0].imshow(U_Array(data_points_NN[0]), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[1, 0].set_title('Heatmap of U(x, z) (Predicted)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z')
    fig.colorbar(imU_NN, ax=axes[1, 0], label='U(x, z)')

    imV_NN = axes[1, 1].imshow(V_Array(data_points_NN[0]), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[1, 1].set_title('Heatmap of V(x, z) (Predicted)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('z')
    fig.colorbar(imV_NN, ax=axes[1, 1], label='V(x, z)')

    imW_NN = axes[1, 2].imshow(W_Array(data_points_NN[0]), extent=(x.min(), x.max(), z.min(), z.max()), origin='lower', cmap='viridis')
    axes[1, 2].set_title('Heatmap of W(x, z) (Predicted)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('z')
    fig.colorbar(imW_NN, ax=axes[1, 2], label='W(x, z)')

    def update(frame):
        imU.set_data(U_Array(data_points[frame]))
        imV.set_data(V_Array(data_points[frame]))
        imW.set_data(W_Array(data_points[frame]))

        imU_NN.set_data(U_Array(data_points_NN[frame]))
        imV_NN.set_data(V_Array(data_points_NN[frame]))
        imW_NN.set_data(W_Array(data_points_NN[frame]))

        return [imU, imV, imW, imU_NN, imV_NN, imW_NN]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False, repeat = False)

    plt.tight_layout()
    plt.show()

    writer = FFMpegWriter(fps=int(1/0.1))

    anim.save('U_V_W_animation.mp4', writer=writer)

    plt.close(fig)

def Predict_During_Time_T(current_frame, time_length):
    predicted_frames = [current_frame]
    predicted_frame = current_frame
    for i in range(time_length):
        predicted_frame = np.expand_dims(np.expand_dims(predicted_frame, axis = 0), axis = 1)
        predicted_frame = cnn_predictor.predict(predicted_frame)[0]
        predicted_frames.append(predicted_frame)
    return predicted_frames

def Predict_After_Time_T (current_frame, time_length):
    predicted_frame = current_frame
    for i in range(time_length):
        predicted_frame = np.expand_dims(np.expand_dims(predicted_frame, axis = 0), axis = 1)
        predicted_frame = cnn_predictor.predict(predicted_frame)[0]
    return predicted_frame

start_time = 0
time_length = 200
Normalized_MSE_Loss_Over_Time(start_time, time_length)
# example = data[start_time]
# example_output = data[start_time+time_length]
# example_output_frames = data[start_time:start_time+time_length+1]
# predicted_frames = Predict_During_Time_T(example, time_length)
# Print_U_V_W_Animation(example_output_frames, predicted_frames, num_frames=time_length+1)

# predicted_example_output = Predict_After_Time_T(example, time_length)
# print(Loss_Of_Frames(data, loss_function="mse"))
# print("\n")
# print(Loss_Of_Dataset(data, loss_function="mse"))
#Print_U_V_W(example_output, predicted_example_output)
