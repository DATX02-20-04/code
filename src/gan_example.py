import tensorflow as tf
import preprocess
from losses import create_simple_gan_loss
from training import create_train_loop, create_gan_train_step
from datasets.nsynth import nsynth_from_tfrecord
from models.simple_gan import create_generator, create_discriminator

# Some compatability options for some graphics cards
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# Setup hyperparameters
hparams = {
    'epochs': 10,
    'steps_per_epoch': 10,
    'sample_rate': 16000,
    'batch_size': 64,
    'buffer_size': 1000,
    'latent_size': 100,
    'gen_lr': 0.0001,
    'disc_lr': 0.0004,
    'log_amin': 1e-5
}

# Load nsynth dataset from a tfrecord
dataset = nsynth_from_tfrecord('/home/big/datasets/nsynth/nsynth-train.tfrecord')

class GANExample():
    def __init__(self, dataset=dataset, hparams=hparams):
        self.dataset = dataset
        self.hparams = hparams
        # Create preprocessing pipeline for the melspectograms
        self.dataset = preprocess.pipeline(self.dataset, [
            preprocess.extract('audio'),
            preprocess.melspec(sr=hparams['sample_rate']),
            preprocess.pad([[0, 0], [0, 4]], 'CONSTANT', constant_values=hparams['log_amin']),
            preprocess.amp_to_log(amin=hparams['log_amin']),
        ])

        # Determine shape of the spectograms in the dataset
        self.spec_shape = None
        for e in self.dataset.take(1):
            self.spec_shape = e.shape
            print(f'Spectogram shape: {spec_shape}')

        # Make sure we got a shape before continuing
        assert self.spec_shape is not None, "Could not get spectogram shape"

        # Make sure the dimensions of spectogram is divisible by 4.
        # This is because the generator is going to upscale it's state twice with a factor of 2.
        assert self.spec_shape[0] % 4 == 0 and self.spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

        # Create preprocessing pipeline for shuffling and batching
        self.dataset = preprocess.pipeline(self.dataset, [
            preprocess.set_channels(1),
            preprocess.shuffle(hparams['buffer_size']),
            preprocess.batch(hparams['batch_size']),
            preprocess.prefetch()
        ])

        # Create the generator and discriminator loss functions
        self.generator_loss, self.discriminator_loss = create_simple_gan_loss(tf.keras.losses.BinaryCrossentropy(from_logits=True))

        # Define the optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(hparams['gen_lr'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(hparams['disc_lr'])

        # Create the actual models for the generator and discriminator
        self.generator = create_generator(hparams['latent_size'], hparams['batch_size'], spec_shape)
        self.discriminator = create_discriminator(spec_shape)

        # Create the GAN train step
        self.train_step = create_gan_train_step(generator,
                                        discriminator,
                                        generator_loss,
                                        discriminator_loss,
                                        generator_optimizer,
                                        discriminator_optimizer,
                                        hparams['batch_size'],
                                        hparams['latent_size'])

        # Define some metrics to be used in the training
        self.gen_loss_avg = tf.keras.metrics.Mean()
        self.disc_loss_avg = tf.keras.metrics.Mean()

        # Create the training loop
        self.train = create_train_loop(self.dataset,
                                self.train_step,
                                on_epoch_start=self.on_epoch_start,
                                on_step=self.on_step,
                                on_epoch_complete=self.on_epoch_complete)


    # This runs at the start of every epoch
    def on_epoch_start(self, epoch, step):
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()

    # This runs at every step in the training (for each batch in dataset)
    def on_step(self, step, stats):
        gen_loss, disc_loss = stats
        gen_loss_avg(gen_loss)
        disc_loss_avg(disc_loss)

    # This runs at the end of every epoch and is used to display metrics
    def on_epoch_complete(self, epoch, step):
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}")
