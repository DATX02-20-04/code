import tensorflow as tf
import math
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class VAEGAN():
    def __init__(self, hparams):
        self.hparams = hparams
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros((self.hparams['latent_size'])), scale=1), reinterpreted_batch_ndims=1)
        self.negloglik = lambda y_pred, y_target: -y_pred.log_prob(y_target)

        self.vaegan, self.encoder, self.decoder = self.create_vaegan()
        self.critic = self.create_critic()
        self.vae_optimizer = tf.keras.optimizers.Adam(self.hparams['lr'])
        self.critic_optimizer = tf.keras.optimizers.Adam(self.hparams['lr'])

    @tf.function
    def train_step(self, x):
        x, _ = x
        epsilon = 0.000001
        with tf.GradientTape() as vae_tape, tf.GradientTape() as critic_tape:
            y = self.vaegan(x, training=True)
            #Can't shuffle directly, see: https://github.com/tensorflow/tensorflow/issues/6269
            random_y = tf.gather(y, tf.random.shuffle(tf.range(tf.shape(y)[0])))
            real_encoded = self.encoder(x)
            y_encoded = self.encoder(x)
            encoding_loss = tf.square(y_encoded-real_encoded)
            [pos_critic, pos_true] = self.critic([x, y])
            [neg_critic, neg_true] = self.critic([y, x])
            [pos_critic2, pos_false] = self.critic([x, random_y])
            [neg_critic2, neg_false] = self.critic([random_y, x])
            similarity_loss = -tf.math.log(1+epsilon-pos_true)-tf.math.log(1+epsilon-neg_true)-tf.math.log(epsilon+pos_false)-tf.math.log(epsilon+neg_false)
            reg_loss = self.encoder.losses[0]
            critic_loss = tf.reduce_mean(-tf.math.log(1+epsilon-((pos_critic + (1 - neg_critic))/2))) \
                        + tf.reduce_mean(-tf.math.log(1+epsilon-((pos_critic2 + (1 - neg_critic2))/2))) \
                        + tf.reduce_mean(similarity_loss)
            vae_loss = tf.reduce_mean(-tf.math.log(1+epsilon-((neg_critic + (1 - pos_critic))/2)) + encoding_loss + reg_loss) \
                     + tf.reduce_mean(similarity_loss)

        vae_gradients = vae_tape.gradient(vae_loss, self.vaegan.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.vae_optimizer.apply_gradients(zip(vae_gradients, self.vaegan.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return vae_loss, critic_loss, similarity_loss

    def create_vaegan(self):
        input_size = self.hparams['window_samples']
        scale = self.hparams['model_scale']
        n_layers = 6

        # Create the encoder
        i = tfkl.Input(shape=(input_size, 1))
        o = i
        for n in range(1,n_layers+1):
            o = tfkl.Conv1D(scale*(n+1), kernel_size=4, strides=2, padding='same')(o)
            o = tfkl.BatchNormalization(axis=1)(o)
            o = tfkl.ReLU()(o)

        o = tfkl.Flatten()(o)

        o = tfkl.Dense(tfpl.IndependentNormal.params_size(self.hparams['latent_size']), activation=None)(o)
        o = tfpl.IndependentNormal(self.hparams['latent_size'], activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=2.0))(o)

        encoder = tfk.Model(inputs=i, outputs=o)

        # Create the decoder
        i = tfkl.Input(shape=(self.hparams['latent_size'],))
        s = scale*(n_layers+1)
        o = tfkl.Dense(s, activation='relu')(i)
        o = tfkl.Reshape(target_shape=(1, s))(o)

        for n in range(1,n_layers+1):
            o = tfkl.UpSampling1D(size=2)(o)
            o = tfkl.Conv1D(scale*(n_layers+1-n), kernel_size=4, strides=1, padding='same')(o)
            o = tfkl.BatchNormalization(axis=1)(o)
            o = tfkl.ReLU()(o)

        o = tfkl.Cropping1D((0,o.shape[1]-input_size))(o)
        o = tfkl.Conv1D(1, kernel_size=1, strides=1, padding='same')(o)
        o = tfkl.Flatten()(o)

        o = tfkl.Dense(tfpl.IndependentNormal.params_size((self.hparams['window_samples'], 1)), activation='tanh')(o)
        o = tfpl.IndependentNormal((self.hparams['window_samples'],1))(o)

        decoder = tfk.Model(inputs=i, outputs=o)

        vaegan = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
        
        return vaegan, encoder, decoder

    def create_critic(self):
        # The critic is used to compute an adversarial-based loss for the
        # vae output. 
        
        input_size = self.hparams['window_samples']
        scale = self.hparams['model_scale']
        n_layers = 6
        
        i1 = tfkl.Input(shape=(input_size, 1))
        i2 = tfkl.Input(shape=(input_size, 1))
        o = tfkl.Concatenate()([i1, i2])

        for n in range(1,n_layers+1):
            o = tfkl.Conv1D(scale*(n+1), kernel_size=4, strides=2, padding='same')(o)
            o = tfkl.BatchNormalization(axis=1)(o)
            o = tfkl.ReLU()(o)

        o = tfkl.Flatten()(o)
        #o1: 0 when real is left, 1 when real is right
        o1 = tfkl.Dense(1, activation='sigmoid')(o)
        #o2: 0 when both represent the same thing, 1 when they represent different things
        o2 = tfkl.Dense(1, activation='sigmoid')(o)

        critic = tfk.Model(inputs=[i1, i2], outputs=[o1,o2])
        return critic


    def sample(self, num_samples):
        zs = tf.random.normal([num_samples, self.hparams['latent_size']])
        output = []
        for z in tf.unstack(zs):
            decoded = tf.reshape(self.decoder(tf.reshape(z, [1, -1]), training=False).mean(), [-1])
            output.append(decoded)

        return tf.concat(output, axis=0)
