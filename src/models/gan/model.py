import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class GAN():
    def __init__(self, shape, hparams):
        super(GAN, self).__init__()
        self.shape = shape
        self.hparams = hparams

        self.generator_optimizer = tfk.optimizers.Adam(self.hparams['gen_lr'])
        self.discriminator_optimizer = tfk.optimizers.Adam(self.hparams['disc_lr'])

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        self.cross_entropy = tfk.losses.BinaryCrossentropy(from_logits=True)
        self.categorical_cross_entropy = tfk.losses.CategoricalCrossentropy()

    @tf.function
    def train_step(self, x):
        noise = tf.random.normal([self.hparams['batch_size'], self.hparams['latent_size']])
        pitches = tf.random.uniform([self.hparams['batch_size']], 0, self.hparams['cond_vector_size'], dtype=tf.int32)
        fake_pitch_index = tf.one_hot(pitches, self.hparams['cond_vector_size'], axis=1)
        # fake_pitch = tf.one_hot(pitches, self.hparams['cond_vector_size'], axis=1)

        real_spec = x['audio']
        real_pitch_index = x['pitch']

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_spec = self.generator([noise, fake_pitch_index], training=True)

            real_output, real_aux = self.discriminator(real_spec, training=True)
            fake_output, fake_aux = self.discriminator(fake_spec, training=True)

            gen_loss = self.generator_loss(fake_output, fake_aux, fake_pitch_index)
            disc_loss = self.discriminator_loss(real_output, fake_output, real_aux, real_pitch_index, fake_aux, fake_pitch_index)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def discriminator_loss(self, real_output, fake_output, real_aux, real_pitch_index, fake_aux, fake_pitch_index):
        real_loss = self.cross_entropy(
            tf.ones_like(real_output)-tf.random.uniform(real_output.shape, 0, 0.1), # Add random to smooth real labels
            real_output)
        fake_loss = self.cross_entropy(
            tf.zeros_like(fake_output)+tf.random.uniform(fake_output.shape, 0, 0.1), # Subtract random to smooth fake labels
            fake_output)
        real_aux_loss = self.categorical_cross_entropy(
            real_pitch_index, real_aux)
        fake_aux_loss = self.categorical_cross_entropy(
            fake_pitch_index, fake_aux)

        total_loss = real_loss + fake_loss + real_aux_loss + fake_aux_loss
        return total_loss

    def generator_loss(self, fake_output, fake_aux, fake_pitch_index):
        fake_aux_loss = self.categorical_cross_entropy(
            fake_pitch_index, fake_aux)
        source_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        return source_loss + fake_aux_loss * self.hparams['aux_loss_weight']

    def create_generator(self):
        latent = tfkl.Input(shape=(self.hparams['latent_size']))
        # One hot vector of size <cond_vector_size>
        pitch_class = tfkl.Input(shape=(self.hparams['cond_vector_size']))

        inp = tfkl.concatenate([latent, pitch_class])
        inp = tfkl.Reshape((1, 1, 161))(inp) # TODO: Fix this to correctly reshape depending on inputs

        o = tfkl.Dense((self.shape[0]//4)*(self.shape[1]//4)*self.hparams['generator_scale'])(inp)
        # o = tfkl.BatchNormalization()(o)
        o = tfkl.LeakyReLU()(o)

        o = tfkl.Reshape((self.shape[0]//4, self.shape[1]//4, self.hparams['generator_scale']))(o)

        o = tfkl.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
        o = tfkl.BatchNormalization()(o)
        o = tfkl.LeakyReLU()(o)

        o = tfkl.UpSampling2D(size=(2, 2))(o)
        o = tfkl.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)(o)
        o = tfkl.BatchNormalization()(o)
        o = tfkl.LeakyReLU()(o)

        o = tfkl.UpSampling2D(size=(2, 2))(o)
        o = tfkl.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(o)


        return tf.keras.Model(inputs=[latent, pitch_class], outputs=o)


    def create_discriminator(self):
        image = tfkl.Input(shape=[*self.shape, 1])

        o = tfkl.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(image)
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
        #
        o = tfkl.Dropout(0.4)(o)
        o = tfkl.Flatten()(o)

        o = tfkl.Dense(32)(o)
        o = tfkl.BatchNormalization()(o)
        o = tfkl.LeakyReLU()(o)

        o = tfkl.Dense(16)(o)
        o = tfkl.BatchNormalization()(o)
        o = tfkl.LeakyReLU()(o)

        fake = tfkl.Dense(1)(o)
        aux = tfkl.Dense(self.hparams['cond_vector_size'], activation='softmax', name='auxillary')(o)

        return tf.keras.Model(inputs=image, outputs=[fake, aux])

