import tensorflow as tf
from tensorflow.keras import layers

def create_generator(latent_size, scale, shape):
    i = layers.Input(shape=(latent_size,))

    o = layers.Dense((shape[0]//4)*(shape[1]//4)*scale)(i)
    o = layers.BatchNormalization()(o)
    o = layers.LeakyReLU()(o)

    o = layers.Reshape((shape[0]//4, shape[1]//4, scale))(o)

    o = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = layers.BatchNormalization()(o)
    o = layers.LeakyReLU()(o)

    o = layers.UpSampling2D(size=(2, 2))(o)
    o = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = layers.BatchNormalization()(o)
    o = layers.LeakyReLU()(o)

    o = layers.UpSampling2D(size=(2, 2))(o)
    o = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = layers.BatchNormalization()(o)
    o = layers.LeakyReLU()(o)

    o = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(o)

    return tf.keras.Model(inputs=i, outputs=o)


def create_discriminator(shape):
    i = layers.Input(shape=[*shape, 1])

    o = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(i)
    o = layers.BatchNormalization()(o)
    o = layers.LeakyReLU()(o)
    # o = layers.Dropout(0.3)(o)

    o = layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same')(o)
    o = layers.BatchNormalization()(o)
    o = layers.LeakyReLU()(o)
    # o = layers.Dropout(0.3)(o)

    # o = layers.Conv2D(8, (4, 4), strides=(2, 2), padding='same')(o)
    # o = layers.BatchNormalization()(o)
    # o = layers.LeakyReLU()(o)

    o = layers.Dropout(0.4)(o)
    o = layers.Flatten()(o)
    o = layers.Dense(1)(o)

    return tf.keras.Model(inputs=i, outputs=o)
