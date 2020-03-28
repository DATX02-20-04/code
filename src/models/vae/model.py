import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import math

class VAE():
    def __init__(self, hparams):
        self.hparams = hparams
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros((self.hparams['latent_size'])), scale=1), reinterpreted_batch_ndims=1)
        self.negloglik = lambda y_pred, y_target: -y_pred.log_prob(y_target)

        self.vae, self.encoder, self.decoder = self.create_vae()
        self.optimizer = tf.keras.optimizers.RMSprop(self.hparams['lr'])

    @tf.function
    def train_step(self, x):
        x, y_target = x
        with tf.GradientTape() as tape:
            y = self.vae(x, training=True)

            reg_loss = self.encoder.losses[0]
            loss = tf.keras.losses.MSE(y, y_target) + reg_loss

        gradients = tape.gradient(loss, self.vae.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))

        return loss

    def create_vae(self):
        input_size = self.hparams['window_samples']
        scale = self.hparams['model_scale']
        n_layers = math.ceil(math.log2(input_size))

        i = tfkl.Input(shape=(input_size, 1))
        o = i
        for n in range(1,n_layers+1):
            o = tfkl.Conv1D(scale*(n+1), kernel_size=2, strides=2, padding='same')(o)
            o = tfkl.BatchNormalization(axis=1)(o)
            o = tfkl.ReLU()(o)

        o = tfkl.Flatten()(o)

        o = tfkl.Dense(tfpl.IndependentNormal.params_size(self.hparams['latent_size']), activation=None)(o)
        o = tfpl.IndependentNormal(self.hparams['latent_size'], activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=2.0))(o)

        encoder = tfk.Model(inputs=i, outputs=o)
        encoder.summary()

        i = tfkl.Input(shape=(self.hparams['latent_size'],))
        s = scale*(n_layers+1)
        o = tfkl.Dense(s, activation='relu')(i)
        o = tfkl.Reshape(target_shape=(1, s))(o)

        for n in range(1,n_layers+1):
            o = tfkl.UpSampling1D(size=2)(o)
            o = tfkl.Conv1D(scale*(n_layers+1-n), kernel_size=2, strides=1, padding='same')(o)
            o = tfkl.BatchNormalization(axis=1)(o)
            o = tfkl.ReLU()(o)

        o = tfkl.Cropping1D((0,o.shape[1]-input_size))(o)
        o = tfkl.Conv1D(1, kernel_size=1, strides=1, padding='same')(o)

        
        decoder = tfk.Model(inputs=i, outputs=o)
        decoder.summary()

        vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
        vae.summary()

        return vae, encoder, decoder

    def sample(self, num_samples):
        zs = tf.random.normal([num_samples, self.hparams['latent_size']])
        output = []
        for z in tf.unstack(zs):
            decoded = tf.reshape(self.decoder(tf.reshape(z, [1, -1]), training=False), [-1])
            output.append(decoded)

        return tf.concat(output, axis=0)
