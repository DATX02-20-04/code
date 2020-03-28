import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

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
            loss = self.negloglik(y, y_target) + reg_loss

        gradients = tape.gradient(loss, self.vae.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))

        return loss

    def create_vae(self):
        i = tfkl.Input(shape=(self.hparams['window_samples'], 1))

        o = tfkl.Conv1D(self.hparams['model_scale'], kernel_size=8, strides=8, dilation_rate=1)(i)
        o = tfkl.BatchNormalization(axis=1)(o)
        o = tfkl.ReLU()(o)

        o = tfkl.Conv1D(2 * self.hparams['model_scale'], kernel_size=3, strides=2)(o)
        o = tfkl.BatchNormalization(axis=1)(o)
        o = tfkl.ReLU()(o)

        o = tfkl.Flatten()(o)

        o = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.hparams['latent_size']), activation=None)(o)
        o = tfpl.MultivariateNormalTriL(self.hparams['latent_size'], activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))(o)

        encoder = tfk.Model(inputs=i, outputs=o)

        s = self.hparams['window_samples'] // 4

        i = tfkl.Input(shape=(self.hparams['latent_size'],))

        o = tfkl.Dense(s, activation='relu')(i)
        o = tfkl.Reshape(target_shape=(s, 1))(o)

        o = tfkl.UpSampling1D(size=2)(o)
        o = tfkl.Conv1D(2 * self.hparams['model_scale'], kernel_size=3, strides=1, padding='SAME')(o)
        o = tfkl.BatchNormalization(axis=1)(o)
        o = tfkl.ReLU()(o)

        o = tfkl.UpSampling1D(size=2)(o)
        o = tfkl.Conv1D(self.hparams['model_scale'], kernel_size=3, strides=1, padding='SAME')(o)
        o = tfkl.BatchNormalization(axis=1)(o)
        o = tfkl.ReLU()(o)

        o = tfkl.Conv1D(1, kernel_size=3, strides=1, padding='SAME')(o)
        o = tfkl.BatchNormalization(axis=1)(o)
        o = tfkl.ReLU()(o)

        o = tfkl.Flatten()(o)

        o = tfkl.Dense(tfpl.IndependentNormal.params_size((self.hparams['window_samples'], 1)), activation='tanh')(o)

        o = tfpl.IndependentNormal((self.hparams['window_samples'],1))(o)

        decoder = tfk.Model(inputs=i, outputs=o)

        vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

        return vae, encoder, decoder

    def sample(self, num_samples):
        zs = tf.random.normal([num_samples, self.hparams['latent_size']])
        output = []
        for z in tf.unstack(zs):
            decoded = tf.reshape(self.decoder(tf.reshape(z, [1, -1]), training=False).mean(), [-1])
            output.append(decoded)

        return tf.concat(output, axis=0)
