import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

tfpl = tfp.layers
tfd = tfp.distributions


def create_gan_vae(shape, latent_size, gen_z_size, scale, prior):
    i = tfkl.Input(shape=[*shape, 1])

    o = tfkl.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(i)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)
    # o = tfkl.Dropout(0.3)(o)

    o = tfkl.Conv2D(16, (4, 4), strides=(2, 2), padding='same')(o)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)
    # o = tfkl.Dropout(0.3)(o)

    # o = tfkl.Conv2D(8, (4, 4), strides=(2, 2), padding='same')(o)
    # o = tfkl.BatchNormalization()(o)
    # o = tfkl.LeakyReLU()(o)

    o = tfkl.Dropout(0.4)(o)
    o = tfkl.Flatten()(o)

    o = tfkl.Dense(tfpl.IndependentNormal.params_size(latent_size), activation="relu")(o)
    o = tfkl.Dense(tfpl.IndependentNormal.params_size(latent_size), activation=None)(o)
    o = tfpl.IndependentNormal(latent_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=2.0))(o)

    encoder = tfk.Model(inputs=i, outputs=o)


    i = tfkl.Input(shape=(latent_size,))

    o = tfkl.Dense((shape[0] // 4) * (shape[1] // 4) * scale)(i)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)

    o = tfkl.Reshape((shape[0] // 4, shape[1] // 4, scale))(o)

    o = tfkl.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)

    o = tfkl.UpSampling2D(size=(2, 2))(o)
    o = tfkl.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)

    o = tfkl.UpSampling2D(size=(2, 2))(o)
    o = tfkl.Conv2D(2, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(o)
    o = tfkl.Flatten()(o)
    o = tfpl.IndependentNormal((*shape,1))(o)

    decoder = tfk.Model(inputs=i, outputs=o)

    vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

    i = tfkl.Input(shape=(latent_size+gen_z_size,))

    o = tfkl.Dense((shape[0] // 4) * (shape[1] // 4) * scale)(i)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)

    o = tfkl.Reshape((shape[0] // 4, shape[1] // 4, scale))(o)

    o = tfkl.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)

    o = tfkl.UpSampling2D(size=(2, 2))(o)
    o = tfkl.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)

    o = tfkl.UpSampling2D(size=(2, 2))(o)
    o = tfkl.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(o)

    generator = tfk.Model(inputs=i, outputs=o)

    i = tfkl.Input(shape=[*shape, 1])

    o = tfkl.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(i)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)
    # o = layers.Dropout(0.3)(o)

    o = tfkl.Conv2D(16, (4, 4), strides=(2, 2), padding='same')(o)
    o = tfkl.BatchNormalization()(o)
    o = tfkl.LeakyReLU()(o)
    # o = layers.Dropout(0.3)(o)

    # o = layers.Conv2D(8, (4, 4), strides=(2, 2), padding='same')(o)
    # o = layers.BatchNormalization()(o)
    # o = layers.LeakyReLU()(o)

    o = tfkl.Dropout(0.4)(o)
    o = tfkl.Flatten()(o)
    o = tfkl.Dense(1, activation='sigmoid')(o)

    discriminator = tfk.Model(inputs=i, outputs=o)

    vae.summary()
    return vae, encoder, decoder, generator, discriminator
