import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import json
import matplotlib.pyplot as plt

class KoopmanNetwork(Model):
    def __init__(self, input_shape, latent_dim):
        super(KoopmanNetwork, self).__init__()
        self.input_shape = input_shape  
        self.flat_dim = np.prod(input_shape) 
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Flatten(), 
            layers.Dense(512, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim, activation=None) 
        ])

        self.K = tf.Variable(tf.random.normal([latent_dim, latent_dim]), trainable=True)

        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.flat_dim, activation=None), 
            layers.Reshape(self.input_shape)  
        ])

    def call(self, x):
        z = self.encoder(x)
        z_next = tf.matmul(z, self.K)
        x_next = self.decoder(z_next)
        return x_next, z, z_next


def explicit_loss(x1, x2, model, alpha1=1.0, alpha2=0.1, alpha3=0.001):
    z1 = model.encoder(x1)
    z2 = model.encoder(x2)

    # Ensuring the eigenfunction is invertible
    x1_recon = model.decoder(z1)
    L_recon = tf.reduce_mean(tf.square(x1 - x1_recon))

    z1_pred = tf.matmul(z1, model.K)
    x1_pred = model.decoder(z1_pred)
    L_pred = tf.reduce_mean(tf.square(x2 - x1_pred))

    # Ensuring Koopman operator is linear
    z2_pred = tf.matmul(z1, model.K)
    L_lin = tf.reduce_mean(tf.square(z2 - z2_pred))

    # L_infinity
    L_inf = tf.reduce_max(tf.abs(x1 - x1_recon)) + tf.reduce_max(tf.abs(x2 - x1_pred))

    # L2 norm of weights
    L_reg = tf.add_n([tf.nn.l2_loss(w) for w in model.trainable_weights])

    total_loss = alpha1 * (L_recon + L_pred) + L_lin + alpha2 * L_inf + alpha3 * L_reg
    return total_loss


def train_koopman_nn(data, input_shape, latent_dim, epochs=150, batch_size=128):
    model = KoopmanNetwork(input_shape, latent_dim)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        for i in range(0, len(data) - 1, batch_size):
            if i + batch_size + 1 > len(data):
                break

            x_batch = data[i:i + batch_size]
            y_batch = data[i + 1:i + 1 + batch_size]

            with tf.GradientTape() as tape:
                loss = explicit_loss(x_batch, y_batch, model)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")

    return model


if __name__ == "__main__":
    H, W, C = 64, 64, 1  #Training only the U velocity for now
    with open('training.json', 'r') as file:
        data = json.load(file)
        data = np.array(data)
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    input_shape = (H, W, C) 
    latent_dim = 100  
    model = train_koopman_nn(data, input_shape, latent_dim)

    x_next_pred, _, _ = model(data[:-1])  
    y_true = data[1:] 

    original = y_true[0].numpy()
    predicted = x_next_pred[0].numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted)
    plt.title("Reconstructed")
    plt.show()
