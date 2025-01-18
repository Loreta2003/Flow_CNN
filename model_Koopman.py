import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

class KoopmanNetwork(Model):
    def __init__(self, input_dim, latent_dim):
        super(KoopmanNetwork, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation=None)  # Latent representation
        ])

        self.K = tf.Variable(tf.random.normal([latent_dim, latent_dim]), trainable=True)

        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation=None)  #Original input
        ])

    def call(self, x):
        z = self.encoder(x)
        z_next = tf.matmul(z, self.K)
        x_next = self.decoder(z_next)
        return x_next, z, z_next

def koopman_loss(y_true, y_pred, z, z_next, K):
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    dynamics_loss = tf.reduce_mean(tf.square(z_next - tf.matmul(z, K)))

    return reconstruction_loss + dynamics_loss

def train_koopman_nn(data, input_dim, latent_dim, epochs=100, batch_size=32):
    model = KoopmanNetwork(input_dim, latent_dim)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        for i in range(0, len(data) - 1, batch_size):
            if i + batch_size + 1 > len(data):
                break  # Drop last incomplete batch

            x_batch = data[i:i+batch_size]
            y_batch = data[i+1:i+1+batch_size]

            with tf.GradientTape() as tape:
                x_next_pred, z, z_next = model(x_batch)
                loss = koopman_loss(y_batch, x_next_pred, z, z_next, model.K)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")

    return model

if __name__ == "__main__":
    t = np.linspace(0, 10, 1000)
    data = np.sin(t).reshape(-1, 1) 
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    input_dim = 1  
    latent_dim = 100 
    model = train_koopman_nn(data, input_dim, latent_dim)

    x_next_pred, _, _ = model(data[:-1])  
    y_true = data[1:] 

    plt.plot(y_true.numpy().flatten(), label="True", linestyle="dashed")
    plt.plot(x_next_pred.numpy().flatten(), label="Predicted")
    plt.legend()
    plt.title("True vs Predicted")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.show()
