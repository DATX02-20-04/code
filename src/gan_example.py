import tensorflow as tf
import preprocess
import os
from losses import create_simple_gan_loss
from training import Trainer, create_gan_train_step
from datasets.nsynth import nsynth_from_tfrecord, instruments
from models.simple_gan import create_generator, create_discriminator
import matplotlib.pyplot as plt
import IPython.display as display

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
    'generator_scale': 256,
    'gen_lr': 0.0001,
    'disc_lr': 0.0004,
    'log_amin': 1e-5,
    'num_examples': 16
}

# Load nsynth dataset from a tfrecord
dataset = nsynth_from_tfrecord('/home/big/datasets/nsynth/nsynth-train.tfrecord')

class GANExample():
    def __init__(self, dataset=dataset, hparams=hparams):
        self.dataset = dataset
        self.hparams = hparams

        # Determine shape of the spectograms in the dataset
        self.spec_shape = None
        for e in self.dataset.take(1):
            self.spec_shape = e.shape
            print(f'Spectogram shape: {self.spec_shape}')

        # Make sure we got a shape before continuing
        assert self.spec_shape is not None, "Could not get spectogram shape"

        # Make sure the dimensions of spectogram is divisible by 4.
        # This is because the generator is going to upscale it's state twice with a factor of 2.
        assert self.spec_shape[0] % 4 == 0 and self.spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

        # Create preprocessing pipeline for shuffling and batching
        self.dataset = preprocess.pipeline(self.dataset, [
            preprocess.set_channels(1),
            preprocess.shuffle(self.hparams['buffer_size']),
            preprocess.batch(self.hparams['batch_size']),
            preprocess.prefetch()
        ])

        # Create the generator and discriminator loss functions
        self.generator_loss, self.discriminator_loss = create_simple_gan_loss(tf.keras.losses.BinaryCrossentropy(from_logits=True))

        # Define the optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(self.hparams['gen_lr'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.hparams['disc_lr'])

        # Create the actual models for the generator and discriminator
        self.generator = create_generator(self.hparams['latent_size'], self.hparams['generator_scale'], self.spec_shape)
        self.discriminator = create_discriminator(self.spec_shape)

        # Define some metrics to be used in the training
        self.gen_loss_avg = tf.keras.metrics.Mean()
        self.disc_loss_avg = tf.keras.metrics.Mean()

        if 'save_dir' in self.hparams:
            self.image_save_dir = os.path.join(self.hparams['save_dir'], 'images/')
            try:
                os.mkdir(self.image_save_dir)
            except:
                pass

        self.seed = tf.random.normal([self.hparams['num_examples'], self.hparams['latent_size']])

        self.trainer = Trainer(self.dataset, self.hparams)

        self.ckpt = tf.train.Checkpoint(
            step=self.trainer.step,
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.generator_optimizer,
            disc_optimizer=self.discriminator_optimizer,
        )

        self.trainer.init_checkpoint(self.ckpt)
        self.trainer.set_train_step(create_gan_train_step(self.generator,
                                                          self.discriminator,
                                                          self.generator_loss,
                                                          self.discriminator_loss,
                                                          self.generator_optimizer,
                                                          self.discriminator_optimizer,
                                                          self.hparams['batch_size'],
                                                          self.hparams['latent_size']))
        self.trainer.on_epoch_start = self.on_epoch_start
        self.trainer.on_step = self.on_step
        self.trainer.on_epoch_complete = self.on_epoch_complete

    def instrument_filter(self, x):
        return tf.reshape(tf.math.equal(x['instrument_family'], instruments[self.hparams['instrument']]), [])

    # This runs at the start of every epoch
    def on_epoch_start(self, epoch, step):
        self.gen_loss_avg = tf.keras.metrics.Mean()
        self.disc_loss_avg = tf.keras.metrics.Mean()

    # This runs at every step in the training (for each batch in dataset)
    def on_step(self, step, stats):
        gen_loss, disc_loss = stats
        self.gen_loss_avg(gen_loss)
        self.disc_loss_avg(disc_loss)
        print(f"Step: {step}, Gen Loss: {self.gen_loss_avg.result()}, Disc Loss: {self.disc_loss_avg.result()}", end='\r')

    # This runs at the end of every epoch and is used to display metrics
    def on_epoch_complete(self, epoch, step, duration):
        display.clear_output(wait=True)
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {self.gen_loss_avg.result()}, Disc Loss: {self.disc_loss_avg.result()}, Duration: {duration} s")
        self.generate_and_save_images_epoch(epoch, step)

    def generate_and_save_images_epoch(self, epoch, step):
        generated = self.generator(self.seed, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(generated.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated[i, :, :, 0])
            plt.axis('off')

        if 'save_images' in self.hparams and self.hparams['save_images']:
            if self.image_save_dir is not None:
                raise Exception("Could not save image, no save_dir was specified in hparams.")
            plt.savefig(os.path.join(self.image_save_dir, 'image_at_epoch_{:04d}_step_{}.png'.format(epoch, step)))

        plt.show()

    def sample_sound(self, seed, pipeline):
        generated = self.generator(seed, training=False)

        print("Generated melspec samples:")
        fig = plt.figure(figsize=(4,4))

        for i in range(generated.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated[i, :, :, 0])
            plt.axis('off')

        plt.show()

        return pipeline(tf.unstack(generated))
