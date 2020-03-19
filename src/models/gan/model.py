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

    @tf.function
    def train_step(self, x):
        noise = tf.random.normal([self.hparams['batch_size'], self.hparams['latent_size']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_x = self.generator(noise, training=True)

            real_output = self.discriminator(x, training=True)
            fake_output = self.discriminator(generated_x, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(
            tf.zeros_like(real_output)+tf.random.uniform(real_output.shape, 0, 0.1), # Add random to smooth real labels
            real_output)
        fake_loss = self.cross_entropy(
            tf.ones_like(fake_output)-tf.random.uniform(fake_output.shape, 0, 0.1), # Subtract random to smooth fake labels
            fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.zeros_like(fake_output), fake_output)

    def create_generator(self):
        i = tfkl.Input(shape=(self.hparams['latent_size'],))

        o = tfkl.Dense((self.shape[0]//4)*(self.shape[1]//4)*self.hparams['generator_scale'])(i)
        o = tfkl.BatchNormalization()(o)
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

        return tf.keras.Model(inputs=i, outputs=o)


    def create_discriminator(self):
        i = tfkl.Input(shape=[*self.shape, 1])

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
        o = tfkl.Dense(1)(o)

        return tf.keras.Model(inputs=i, outputs=o)
