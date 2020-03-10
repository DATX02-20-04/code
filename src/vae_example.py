import tensorflow as tf
import tensorflow_probability as tfp
import preprocess
import os
from losses import create_simple_gan_loss
from training import Trainer, create_gan_train_step
from datasets.maestro import maestro_from_files
from models.vae import create_vae
import matplotlib.pyplot as plt
#import IPython.display as display
from model import Model
import scipy.io.wavfile as wavfile

tfpl = tfp.layers
tfd = tfp.distributions

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class VAEExample(Model):
    def __init__(self, dataset, hparams):
        super(VAEExample, self).__init__(dataset, hparams)

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros((self.hparams['latent_size'])), scale=1), reinterpreted_batch_ndims=1)
        self.negloglik = lambda x, rv_x: -rv_x.log_prob(x)
        self.vae, self.encoder, self.decoder = create_vae(self.hparams['latent_size'], self.hparams['model_scale'], self.hparams['window_samples'], self.prior)
        self.optimizer = tf.keras.optimizers.SGD(self.hparams['lr'], decay=1e-4)

        self.vae.compile(optimizer=self.optimizer, loss=self.negloglik)

        self.seed = tf.random.normal([self.hparams['num_examples'], self.hparams['latent_size']])

    def train(self):
        self.on_epoch_end(0, None)
        cb = tf.keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=self.on_epoch_end, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
        self.vae.fit(self.dataset, epochs=self.hparams['epochs'], validation_data=self.dataset.take(100), steps_per_epoch=self.hparams['steps_per_epoch'], callbacks=[cb])

    def on_epoch_end(self, epoch, logs):
        print("Sampling...")
        z = self.prior.sample(100)
        decoded = self.decoder(z, training=False).mean()
        print("DECODED: ", decoded)
        sample = tf.reshape(decoded, [-1]).numpy()

        print("Plotting...")
        plt.plot(sample)
        plt.show()
        wavfile.write(f'sample{epoch}.wav', self.hparams['sample_rate'], sample)
        print("Done")

    def preprocess(self, dataset):
        return preprocess.pipeline(dataset, [
            preprocess.set_channels(1),
            preprocess.dupe(),
            preprocess.shuffle(self.hparams['buffer_size']),
            preprocess.batch(self.hparams['batch_size']),
            preprocess.prefetch()
        ])



if __name__ == '__main__':
    # Setup hyperparameters
    hparams = {
        'epochs': 10,
        'steps_per_epoch': 2000,
        'sample_rate': 16000,
        'window_samples': 8000,
        'batch_size': 64,
        'buffer_size': 1000,
        'latent_size': 100,
        'model_scale': 32,
        'lr': 1e-3,
        'log_amin': 1e-5,
        'num_examples': 1,
        'save_dir': '.'
    }

    # Load nsynth dataset from a tfrecord
    dataset = maestro_from_files('/home/big/datasets/maestro-v2.0.0/', hparams['window_samples'])

    vae = VAEExample(dataset, hparams)
    vae.train()
