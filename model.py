import tensorflow as tf
from tensorflow.keras import layers, models

dim = 64
input_shape = (None, dim, dim, 3) 
sequence_length = 10 

def build_cnn_predictor(input_shape):
    model = models.Sequential()
    
    model.add(layers.Conv3D(dim, (3, 3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling3D((1, 2, 2), padding='same'))
    
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((1, 2, 2), padding='same'))
    
    model.add(layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((1, 2, 2), padding='same'))
    
    model.add(layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same'))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(dim * dim * 3, activation='tanh')) 
    
    model.add(layers.Reshape((dim, dim, 3)))

    return model
