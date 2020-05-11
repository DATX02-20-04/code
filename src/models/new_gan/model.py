import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import models.new_gan.layers as l
from util import get_plot_image

"""
Based on https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
"""
class GAN(tfk.Model):
    def __init__(self, hparams, stats):
        super(GAN, self).__init__()
        self.hparams = hparams
        self.stats = stats
        self.generator_optimizer = tfk.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8)
        self.discriminator_optimizer = tfk.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8)

        self.cross_entropy = tfk.losses.BinaryCrossentropy(from_logits=True)
        self.categorical_cross_entropy = tfk.losses.CategoricalCrossentropy()

        self.generators = self.create_generator()
        self.discriminators = self.create_discriminator()

    def fake_loss(self, y_true, y_pred):
        # return tf.math.reduce_mean(y_true * y_pred)
        return tfk.losses.mean_squared_error(y_true, y_pred)

    def aux_loss(self, y_true, y_pred):
        return tfk.losses.categorical_crossentropy(y_true, y_pred)

    def get_initial_models(self):
        return self.generators[0][0], self.discriminators[0][0]

    @tf.function
    def generate_pitch(self, samples):
        fake_pitch_index = tf.random.uniform([samples], 0, self.hparams['pitches'], dtype=tf.int32)
        return tf.one_hot(fake_pitch_index, self.hparams['pitches'], axis=1)


    @tf.function
    def generate_fake(self, generator, samples, training=True):
        z = tf.random.normal([samples, self.hparams['latent_dim']])
        y_fake_aux = self.generate_pitch(samples)

        X_fake = generator([z, y_fake_aux], training=training)

        return X_fake, y_fake_aux

    def train_epochs(self, generator, discriminator, dataset, epochs, batch_size, block, step, fade=False, tsw=None):
        half_batch = batch_size // 2
        bpe = self.stats['examples'] // half_batch
        steps = bpe * epochs
        init_step = step.numpy()

        dataset = dataset.batch(half_batch, drop_remainder=True)
        dataset = dataset.map(lambda X_real, pitch_real: (X_real, tf.ones([half_batch, 1]), pitch_real))

        train_step = self.create_train_step(generator, discriminator, half_batch)

        for e in range(epochs):
            for X_real, y_real, y_real_aux in dataset:
                alpha = 0.0
                if fade:
                    alpha = self.update_fadein([generator, discriminator], step.numpy()-init_step, steps)

                d_real_loss, d_fake_loss, d_real_aux_loss, d_fake_aux_loss, d_total_loss, g_aux_loss, g_source_loss, g_total_loss = train_step(X_real, y_real, y_real_aux)

                print(f"e{e}, dr={d_real_loss:.3f}, df={d_fake_loss:.3f} draux={d_real_aux_loss:.3f} dfaux={d_fake_aux_loss:.3f} gsrc={g_source_loss:.3f} gaux={g_aux_loss:.3f} a={alpha:.3f}", end='\r')

                # Save statistics
                if tsw is not None:
                    tb_block = block.numpy()
                    tb_step = step.numpy()

                    with tf.name_scope(f'Block {tb_block}') as scope:
                        with tsw.as_default():
                            tf.summary.scalar('Gen. source loss', g_source_loss, step=tb_step)
                            tf.summary.scalar('Gen. aux loss', g_aux_loss, step=tb_step)
                            tf.summary.scalar('Gen. total loss', g_total_loss, step=tb_step)
                            tf.summary.scalar('Disc. fake loss', d_fake_loss, step=tb_step)
                            tf.summary.scalar('Disc. real loss', d_real_loss, step=tb_step)
                            tf.summary.scalar('Disc. real aux loss', d_real_aux_loss, step=tb_step)
                            tf.summary.scalar('Disc. fake aux loss', d_fake_aux_loss, step=tb_step)
                            tf.summary.scalar('Disc. total loss', d_total_loss, step=tb_step)

                step.assign_add(1)

    def create_train_step(self, generator, discriminator, half_batch):
        @tf.function
        def apply_grad(X_real, y_real, y_real_aux):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                X_fake, y_fake_aux = self.generate_fake(generator, half_batch)

                Y_real, Y_real_aux = discriminator(X_real, training=True)
                Y_fake, Y_fake_aux = discriminator(X_fake, training=True)

                d_real_loss = self.cross_entropy(tf.ones_like(Y_real), Y_real)
                d_fake_loss = self.cross_entropy(tf.zeros_like(Y_fake), Y_fake)

                d_real_aux_loss = self.categorical_cross_entropy(y_real_aux, Y_real_aux)
                d_fake_aux_loss = self.categorical_cross_entropy(y_fake_aux, Y_fake_aux)
                d_total_loss = d_real_loss - (d_fake_loss + d_fake_aux_loss) + d_real_aux_loss

                g_aux_loss = self.categorical_cross_entropy(y_fake_aux, Y_fake_aux)
                g_source_loss = self.cross_entropy(tf.ones_like(X_fake), X_fake)
                g_total_loss = g_source_loss - g_aux_loss * self.hparams['aux_loss_weight']

            gradients_of_generator = gen_tape.gradient(g_total_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(d_total_loss, discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            return d_real_loss, d_fake_loss, d_real_aux_loss, d_fake_aux_loss, d_total_loss, g_aux_loss, g_source_loss, g_total_loss
        return apply_grad


    def add_dblock(self, old, n_layers=3):
        init = tfk.initializers.RandomNormal(stddev=0.02)
        const = tfk.constraints.max_norm(1.0)
        in_shape = list(old.input.shape)
        input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
        in_image = tfkl.Input(shape=input_shape)

        d = tfkl.Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
        d = tfkl.LeakyReLU(alpha=0.2)(d)
        d = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = tfkl.LeakyReLU(alpha=0.2)(d)
        d = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = tfkl.LeakyReLU(alpha=0.2)(d)
        d = tfkl.AveragePooling2D(pool_size=(2, 2))(d)

        block_new = d
        for i in range(n_layers, len(old.layers)-2):
            d = old.layers[i](d)

        d1 = old.layers[-2](d)
        d2 = old.layers[-1](d)

        model1 = tfk.Model(in_image, outputs=[d1, d2])
        downsample = tfkl.AveragePooling2D(pool_size=(2, 2))(in_image)
        block_old = old.layers[1](downsample)
        block_old = old.layers[2](block_old)
        d = l.WeightedSum()([block_old, block_new])

        for i in range(n_layers, len(old.layers)-2):
            d = old.layers[i](d)

        d1 = old.layers[-2](d)
        d2 = old.layers[-1](d)

        model2 = tfk.Model(in_image, outputs=[d1, d2])

        return [model1, model2]

    def create_discriminator(self, input_shape=(4, 32, 1)):
        init = tfk.initializers.RandomNormal(stddev=0.02)
        const = tfk.constraints.max_norm(1.0)
        model_list = []
        in_image = tfkl.Input(shape=input_shape)

        d = tfkl.Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
        d = tfkl.LeakyReLU(alpha=0.2)(d)
        d = l.BatchStd()(d)
        d = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = tfkl.LeakyReLU(alpha=0.2)(d)
        d = tfkl.Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = tfkl.LeakyReLU(alpha=0.2)(d)
        d = tfkl.Flatten()(d)

        # d = tfkl.Dropout(rate=0.2)(d)
        out_fake = tfkl.Dense(1)(d)
        out_aux = tfkl.Dense(self.hparams['pitches'])(d)

        model = tfk.Model(inputs=in_image, outputs=[out_fake, out_aux])
        model_list.append([model, model])

        for i in range(1, self.hparams['n_blocks']):
            old = model_list[i - 1][0]

            models = self.add_dblock(old)
            model_list.append(models)

        return model_list

    def add_gblock(self, old):
        init = tfk.initializers.RandomNormal(stddev=0.02)
        const = tfk.constraints.max_norm(1.0)
        block_end = old.layers[-2].output
        upsampling = tfkl.UpSampling2D()(block_end)

        g = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
        g = l.PixelNorm()(g)
        g = tfkl.LeakyReLU(alpha=0.2)(g)
        g = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = l.PixelNorm()(g)
        g = tfkl.LeakyReLU(alpha=0.2)(g)

        out_image = tfkl.Conv2D(1, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        model1 = tfk.Model(inputs=old.input, outputs=out_image)
        out_old = old.layers[-1]
        out_image2 = out_old(upsampling)
        merged = l.WeightedSum()([out_image2, out_image])
        model2 = tfk.Model(inputs=old.input, outputs=merged)

        return [model1, model2]

    def create_generator(self, in_dim=(4, 32)):
        init = tfk.initializers.RandomNormal(stddev=0.02)
        const = tfk.constraints.max_norm(1.0)
        model_list = []
        in_latent = tfkl.Input(shape=(self.hparams['latent_dim'],))
        in_pitch = tfkl.Input(shape=(self.hparams['pitches'],))

        g = tfkl.Concatenate()([in_latent, in_pitch])

        g = tfkl.Dense(128 * in_dim[0] * in_dim[1], kernel_initializer=init, kernel_constraint=const)(g)
        g = tfkl.Reshape((in_dim[0], in_dim[1], 128))(g)
        g = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = l.PixelNorm()(g)
        g = tfkl.LeakyReLU(alpha=0.2)(g)
        g = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = l.PixelNorm()(g)
        g = tfkl.LeakyReLU(alpha=0.2)(g)

        out_image = tfkl.Conv2D(1, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        model = tfk.Model(inputs=[in_latent, in_pitch], outputs=out_image)
        model_list.append([model, model])

        for i in range(1, self.hparams['n_blocks']):
            old = model_list[i - 1][0]
            models = self.add_gblock(old)
            model_list.append(models)

        return model_list

    def update_fadein(self, models, step, n_steps):
        alpha = step / float(n_steps - 1)

        for model in models:
            for layer in model.layers:
                if isinstance(layer, l.WeightedSum):
                    layer.alpha = alpha

        return alpha

