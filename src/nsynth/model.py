import tensorflow as tf
from tensorflow.keras import layers
from tcn import TCN

def create_model(timesteps, channels):
    i = layers.Input(shape=(timesteps, channels))
    o = TCN(return_sequences=False)(i)
    o = layers.Dense(1, activation='tanh')(o)
    return tf.keras.Model(inputs=i, outputs=o)

