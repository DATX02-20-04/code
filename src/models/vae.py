import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

tfpl = tfp.layers
tfd = tfp.distributions


def create_vae(latent_size, scale, samples, prior):
    i = tfkl.Input(shape=(samples, 1))

    o = tfkl.Conv1D(scale, kernel_size=8, strides=8, dilation_rate=1, activation='relu')(i)

    o = tfkl.Conv1D(2 * scale, kernel_size=3, strides=2, activation='relu')(o)

    o = tfkl.Flatten()(o)

    o = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(latent_size), activation=None)(o)
    o = tfpl.MultivariateNormalTriL(latent_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior))(o)

    encoder = tfk.Model(inputs=i, outputs=o)

    s = samples // 4

    i = tfkl.Input(shape=(latent_size,))

    o = tfkl.Dense(s, activation='relu')(i)
    o = tfkl.Reshape(target_shape=(s, 1))(o)

    o = tfkl.UpSampling1D(size=2)(o)
    o = tfkl.Conv1D(2 * scale, kernel_size=3, strides=1, padding='SAME', activation='relu')(o)

    o = tfkl.UpSampling1D(size=2)(o)
    o = tfkl.Conv1D(scale, kernel_size=3, strides=1, padding='SAME', activation='relu')(o)

    o = tfkl.Conv1D(1, kernel_size=3, strides=1, padding='SAME', activation='relu')(o)

    o = tfkl.Flatten()(o)

    o = tfkl.Dense(tfpl.IndependentNormal.params_size((samples, 1)), activation='tanh')(o)

    o = tfpl.IndependentNormal((samples,1))(o)

    decoder = tfk.Model(inputs=i, outputs=o)

    vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

    return vae, encoder, decoder
