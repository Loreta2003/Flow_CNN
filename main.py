import tensorflow as tf
from tensorflow.keras import layers, models
import model as md
import data as data
import numpy as np

dim = md.dim
sequence_length = md.sequence_length

X_train, X_val = np.array(data.training_data), np.array(data.validation_data)
y_train, y_val = np.array(data.training_output), np.array(data.validation_output)

cnn_predictor = md.build_cnn_predictor(input_shape=(sequence_length, dim, dim, 3))

cnn_predictor.compile(optimizer='adam', loss='mean_squared_error')

cnn_predictor.summary()

history = cnn_predictor.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_val, y_val), verbose=1)
