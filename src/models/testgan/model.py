import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class TESTGAN():
    def __init__(self, shape, hparams):
        super(TESTGAN, self).__init__()
        self.shape = shape
        self.hparams = hparams

        self.generator_optimizer = tfk.optimizers.Adam(self.hparams['gen_lr'])
        self.selector_optimizer = tfk.optimizers.SGD(self.hparams['disc_lr'])

        self.generator = self.create_generator()
        self.selector = self.create_selector()

        self.cross_entropy = tfk.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, x):
        noise = tf.random.normal([self.hparams['batch_size'], self.hparams['latent_size']])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as sel_tape:
            generated_x = self.generator(noise, training=True)

            # when the first argument is real, a positive value should be output
            positive_selection = self.selector([x, generated_x])
            
            # when the second argument is real, a negative value should be output
            negative_selection = self.selector([generated_x, x])

            gen_loss = self.generator_loss(positive_selection, negative_selection)
            sel_loss = self.selector_loss(positive_selection, negative_selection)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_selector = sel_tape.gradient(sel_loss, self.selector.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        #self.selector_optimizer.apply_gradients(zip(gradients_of_selector, self.selector.trainable_variables))

        return gen_loss, sel_loss

    def generator_loss(self, positive_selection, negative_selection):
        # The discriminator wants to confuse the selector as much as possible
        return tf.reduce_mean([tf.square(positive_selection), tf.square(negative_selection)])

    def selector_loss(self, positive_selection, negative_selection):
        # the selector wants to minimize negative_selection and amximize positive_selection
        return tf.reduce_mean(negative_selection) - tf.reduce_mean(positive_selection)

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


    def create_selector(self):
        i = tfkl.Input(shape=[*self.shape, 1])
        i2 = tfkl.Input(shape=[*self.shape, 1])

        o = tfkl.Concatenate()([i,i2])

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

        return tf.keras.Model(inputs=[i,i2], outputs=o)
