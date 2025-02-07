import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import imageio_ffmpeg
import shutil
import matplotlib as mpl

# Ensure FFmpeg is available
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"  # Manually set your FFmpeg path
    if not os.path.exists(ffmpeg_path):
        raise RuntimeError("FFmpeg not found! Ensure it's installed and accessible in your environment.")
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Set FFmpeg path in Matplotlib
mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

class ConvAutoencoder(Model):
    def __init__(self, **kwargs):
        super(ConvAutoencoder, self).__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(64, 64, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def get_config(self):
        config = super(ConvAutoencoder, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def train_conv_autoencoder(input_data, output_data, epochs=10, batch_size=64):
    model = ConvAutoencoder()
    model.compile(optimizer='adam', loss='mse')
    model.fit(input_data, output_data, epochs=epochs, batch_size=batch_size, shuffle=True)
    return model

def Prediction_t_time(initial_input, time, model):
    predicted_output = []
    current = tf.expand_dims(initial_input, axis=0)  # Ensure input is a single sample with batch dim
    for _ in range(len(time)):
        current = model(current)
        predicted_output.append(current.numpy().squeeze())
    return np.array(predicted_output)

if __name__ == "__main__":
    with open('training.json', 'r') as file:
        data = json.load(file)
        data = np.array(data)
        input = data[:-1]
        output = data[1:]
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    output = tf.convert_to_tensor(output, dtype=tf.float32)

    with open('validation.json', 'r') as file:
        valid_data = json.load(file)
        valid_data = np.array(valid_data)
    valid_data = tf.convert_to_tensor(valid_data, dtype=tf.float32)

    model_save_path = 'conv_autoencoder_model.keras'  # Updated file extension

    # Check if model already exists
    if os.path.exists(model_save_path):
        print("Loading pre-trained model...")
        model = tf.keras.models.load_model(model_save_path, custom_objects={'ConvAutoencoder': ConvAutoencoder})
    else:
        print("Training new model...")
        model = train_conv_autoencoder(input, output)
        model.save(model_save_path)  # Saves in native Keras format

    sample_input = valid_data[600]
    true_output = valid_data[601:621]
    predicted_output = Prediction_t_time(sample_input, range(20), model)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax1, ax2 = axes
    img1 = ax1.imshow(true_output[0])
    img2 = ax2.imshow(predicted_output[0])
    ax1.set_title("True Output")
    ax2.set_title("Predicted Output")
    ax1.axis('off')
    ax2.axis('off')

    def update(frame):
        img1.set_array(true_output[frame])
        img2.set_array(predicted_output[frame])
        return [img1, img2]

    writer = animation.FFMpegWriter(fps=10)
    ani = animation.FuncAnimation(fig, update, frames=len(predicted_output), interval=200, blit=True)
    ani.save('predicted_vs_true_animation.mp4', writer=writer)
    plt.show()
