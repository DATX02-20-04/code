import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import models.new_gan.layers as l

"""
Based on https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
"""
class GAN(tfk.Model):
    def __init__(self, hparams, stats, init_size):
        super(GAN, self).__init__()
        self.hparams = hparams
        self.stats = stats
        self.init_size = init_size
        self.optimizer = tfk.optimizers.Adam(lr=hparams['lr'], beta_1=0, beta_2=0.99, epsilon=10e-8)

        self.generators = self.create_generator(init_size)
        self.discriminators = self.create_discriminator([*init_size, 2])
        self.models = self.create_composite(self.discriminators, self.generators)

    def wasserstein_loss(self, y_true, y_pred):
        # return tf.math.reduce_mean(y_true * y_pred)
        return tfk.losses.mean_squared_error(y_true, y_pred)

    def get_initial_models(self):
        return self.generators[0][0], self.discriminators[0][0], self.models[0][0]

    def generate_fake(self, generator, samples):
        z = tf.random.normal([samples, self.hparams['latent_dim']])
        X = generator(z)
        y = tf.zeros([samples, 1])
        return X, y

    def train_epochs(self, generator, discriminator, model, dataset, epochs, batch_size, fade=False):
        half_batch = batch_size // 2
        bpe = self.stats['examples'] // half_batch
        steps = bpe * epochs

        dataset = dataset.batch(half_batch, drop_remainder=True)
        dataset = dataset.map(lambda *x: (x, tf.ones([half_batch, 1])))

        step = 0
        for e in range(epochs):
            for (X_real, pitch_real), y_real in dataset:
                alpha = 0.0
                if fade:
                    alpha = self.update_fadein([generator, discriminator, model], step, steps)

                X_fake, y_fake = self.generate_fake(generator, half_batch)

                d_loss1 = discriminator.train_on_batch(X_real, y_real)
                d_loss2 = discriminator.train_on_batch(X_fake, y_fake)

                z_input = tf.random.normal([batch_size, self.hparams['latent_dim']])
                y_real2 = tf.ones([batch_size, 1])
                g_loss = model.train_on_batch(z_input, y_real2)

                print(f"e{e}, {int((step/steps)*100)}%, {step+1}/{steps}, dr={d_loss1:.3f}, df={d_loss2:.3f} g={g_loss:.3f} a={alpha:.3f}", end='\r')

                step += 1


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
        for i in range(n_layers, len(old.layers)):
            d = old.layers[i](d)

        model1 = tfk.Model(in_image, d)
        model1.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
        downsample = tfkl.AveragePooling2D(pool_size=(2, 2))(in_image)
        block_old = old.layers[1](downsample)
        block_old = old.layers[2](block_old)
        d = l.WeightedSum()([block_old, block_new])

        for i in range(n_layers, len(old.layers)):
            d = old.layers[i](d)

        model2 = tfk.Model(in_image, d)
        model2.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)

        return [model1, model2]

    def create_discriminator(self, input_shape=(16, 8, 2)):
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

        out_class = tfkl.Dense(1)(d)
        model = tfk.Model(in_image, out_class)
        model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
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

        out_image = tfkl.Conv2D(2, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        model1 = tfk.Model(old.input, out_image)
        out_old = old.layers[-1]
        out_image2 = out_old(upsampling)
        merged = l.WeightedSum()([out_image2, out_image])
        model2 = tfk.Model(old.input, merged)

        return [model1, model2]

    def create_generator(self, in_dim=(16, 8)):
        init = tfk.initializers.RandomNormal(stddev=0.02)
        const = tfk.constraints.max_norm(1.0)
        model_list = []
        in_latent = tfkl.Input(shape=(self.hparams['latent_dim'],))

        g = tfkl.Dense(128 * in_dim[0] * in_dim[1], kernel_initializer=init, kernel_constraint=const)(in_latent)
        g = tfkl.Reshape((in_dim[0], in_dim[1], 128))(g)
        g = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = l.PixelNorm()(g)
        g = tfkl.LeakyReLU(alpha=0.2)(g)
        g = tfkl.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = l.PixelNorm()(g)
        g = tfkl.LeakyReLU(alpha=0.2)(g)

        out_image = tfkl.Conv2D(2, (1,1), padding='same', activation='tanh', kernel_initializer=init, kernel_constraint=const)(g)
        model = tfk.Model(in_latent, out_image)
        model_list.append([model, model])

        for i in range(1, self.hparams['n_blocks']):
            old = model_list[i - 1][0]
            models = self.add_gblock(old)
            model_list.append(models)

        return model_list

    def create_composite(self, discriminators, generators):
        model_list = []

        for i in range(len(discriminators)):
            g_models, d_models = generators[i], discriminators[i]

            d_models[0].trainable = False
            model1 = tfk.Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            model1.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)

            d_models[1].trainable = False
            model2 = tfk.Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
           
            model_list.append([model1, model2])
        return model_list

    def update_fadein(self, models, step, n_steps):
        alpha = step / float(n_steps - 1)

        for model in models:
            for layer in model.layers:
                if isinstance(layer, l.WeightedSum):
                    layer.alpha = alpha

        return alpha
