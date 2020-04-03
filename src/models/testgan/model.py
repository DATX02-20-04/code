import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class TESTGAN():
    def __init__(self, shape, hparams):
        super(TESTGAN, self).__init__()
        self.shape = shape
        self.hparams = hparams

        self.generator_optimizer = tfk.optimizers.Adam(self.hparams['gen_lr'])
        # We don't want generator vulnerabilities to be exploited too fast since
        # it leads to unmanagable loss sizes. Therefore we use an optimizer
        # wihtout momentum, like SGD.         
        self.selector_optimizer = tfk.optimizers.Adam(self.hparams['disc_lr'])

        self.generator = self.create_generator()
        self.selector = self.create_selector()

        self.cross_entropy = tfk.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, x):
        batch_size = x['audio'].shape[0]
        noise = tf.random.normal([batch_size, self.hparams['latent_size']])

        real_spec = x['audio']
        real_pitch = x['pitch']

        with tf.GradientTape() as gen_tape, tf.GradientTape() as sel_tape:
            fake_spec = self.generator([noise, real_pitch], training=True)

            positive_selection = self.selector([real_spec, fake_spec, real_pitch], training=True)
            negative_selection = self.selector([fake_spec, real_spec, real_pitch], training=True)

            gen_loss = self.generator_loss(positive_selection, negative_selection)
            sel_loss = self.selector_loss(positive_selection, negative_selection)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_selector = sel_tape.gradient(sel_loss, self.selector.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.selector_optimizer.apply_gradients(zip(gradients_of_selector, self.selector.trainable_variables))

        return gen_loss, sel_loss

    def selector_loss(self, positive_selection, negative_selection):
        # the selector wants to minimize negative_selection and maximize positive_selection
        return tf.reduce_mean(negative_selection) - tf.reduce_mean(positive_selection)

    def generator_loss(self, positive_selection, negative_selection):
        # The discriminator wants to confuse the selector as much as possible
        return tf.reduce_mean([tf.square(positive_selection), tf.square(negative_selection)])

    def create_generator(self):
        i = tfkl.Input(shape=(self.hparams['latent_size'],), name='latent_input')
        cond = tfkl.Input(shape=(self.hparams['cond_vector_size']), name='cond_input')

        o = tfkl.Concatenate(name='latent_cond_concat')([i, cond])

        o = tfkl.Dense((self.shape[0]//4)*(self.shape[1]//4)*self.hparams['generator_scale'], name='fst_dense')(i)
        o = tfkl.BatchNormalization(name='fst_bn')(o)
        o = tfkl.LeakyReLU(name='fst_relu')(o)

        o = tfkl.Reshape((self.shape[0]//4, self.shape[1]//4, self.hparams['generator_scale']), name='reshape')(o)

        o = tfkl.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, name='fst_c2d')(o)
        o = tfkl.BatchNormalization(name='snd_bn')(o)
        o = tfkl.LeakyReLU(name='snd_relu')(o)

        o = tfkl.UpSampling2D(size=(2, 2), name='fst_us')(o)
        o = tfkl.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False, name='snd_c2d')(o)
        o = tfkl.BatchNormalization(name='thd_bn')(o)
        o = tfkl.LeakyReLU(name='thd_relu')(o)

        o = tfkl.UpSampling2D(size=(2, 2), name='snd_us')(o)
        o = tfkl.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh', name='thd_c2d')(o)

        return tf.keras.Model(inputs=[i, cond], outputs=o, name='generator')


    def create_selector(self):
        ai = tfkl.Input(shape=[*self.shape, 1], name='spec_a_input')
        bi = tfkl.Input(shape=[*self.shape, 1], name='spec_b_input')
        o = tfkl.Concatenate(name='spec_concat')([ai,bi])

        o = tfkl.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='fst_c2d')(o)
        o = tfkl.BatchNormalization(name='fst_bn')(o)
        o = tfkl.LeakyReLU(name='fst_lrelu')(o)
        # o = tfkl.Dropout(0.3)(o)

        o = tfkl.Conv2D(16, (4, 4), strides=(2, 2), padding='same', name='snd_c2d')(o)
        o = tfkl.BatchNormalization(name='snd_bn')(o)
        o = tfkl.LeakyReLU(name='snd_lrelu')(o)
        # o = tfkl.Dropout(0.3)(o)

        # o = tfkl.Conv2D(8, (4, 4), strides=(2, 2), padding='same')(o)
        # o = tfkl.BatchNormalization()(o)
        # o = tfkl.LeakyReLU()(o)
        
        cond = tfkl.Input(shape=(self.hparams['cond_vector_size'],), name='cond_input')

        o = tfkl.Dropout(0.4, name='fst_dropout')(o)
        o = tfkl.Flatten(name='flatten')(o)
        o = tfkl.Concatenate(name='concat_cond')([o, cond])

        o = tfkl.Dense(32, name='fst_dense')(o)
        o = tfkl.BatchNormalization(name='thd_bn')(o)
        o = tfkl.LeakyReLU(name='thd_lrelu')(o)

        o = tfkl.Dense(16, name='snd_dense')(o)
        o = tfkl.BatchNormalization(name='bn_4')(o)
        o = tfkl.LeakyReLU(name='lrelu_4')(o)

        o = tfkl.Dense(1, activation='tanh' ,name='final')(o)

        return tf.keras.Model(inputs=[ai, bi, cond], outputs=o, name='selector')
