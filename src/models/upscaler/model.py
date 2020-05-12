import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class Upscaler(tfk.Model):
    def __init__(self, hparams, stats):
        super(Upscaler, self).__init__()
        self.hparams = hparams
        self.stats = stats
        self.optimizer = tfk.optimizers.Adam(lr=0.001)

        self.model = self.create_model()


    def loss(self, y_true, y_pred):
        # return tf.math.reduce_mean(y_true * y_pred)
        return tfk.losses.mean_squared_error(y_true, y_pred)

    def train_epochs(self, dataset, epochs, batch_size):
        bpe = self.stats['examples'] // batch_size
        steps = bpe * epochs


        step = 0
        for e in range(epochs):
            for X, Y in dataset:
                loss = self.model.train_on_batch(X, Y)

                print(f"e{e}, {int((step/steps)*100)}%, {step+1}/{steps}, loss={loss:.3f}", end='\r')

                step += 1


    def create_model(self, in_dim=(32, 256, 2)):
        # init = tfk.initializers.RandomNormal(stddev=0.02)
        # const = tfk.constraints.max_norm(1.0)
        i = tfkl.Input(shape=in_dim)

        g = i
        for _ in range(self.hparams['n_blocks']):
            g = tfkl.UpSampling2D()(g)
            g = tfkl.Conv2D(64, (3,3), padding='same')(g)
            # g = l.PixelNorm()(g)
            g = tfkl.LeakyReLU(alpha=0.2)(g)
            # g = tfkl.UpSampling2D()(i)
            g = tfkl.Conv2D(32, (3,3), padding='same')(g)
            # g = l.PixelNorm()(g)
            g = tfkl.LeakyReLU(alpha=0.2)(g)

        out_image = tfkl.Conv2D(2, (1,1), padding='same')(g)

        model = tfk.Model(i, out_image)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()

        return model

