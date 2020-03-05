import tensorflow as tf
import preprocess
import os
from losses import create_simple_gan_loss
from training import create_train_loop, create_gan_train_step
from datasets.nsynth import nsynth_from_tfrecord
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
    'gen_lr': 0.0001,
    'disc_lr': 0.0004,
    'log_amin': 1e-5,
    'num_examples': 16
}

# Load nsynth dataset from a tfrecord
dataset = nsynth_from_tfrecord('/home/big/datasets/nsynth/nsynth-train.tfrecord')

class GANExample():
    def __init__(self, dataset=dataset, hparams=hparams, save_dir='', save_images=False):
        self.dataset = dataset
        self.hparams = hparams
        self.save_dir = save_dir
        self.save_images = save_images

        # Create preprocessing pipeline for the melspectograms
        self.dataset = preprocess.pipeline(self.dataset, [
            preprocess.extract('audio'),
            preprocess.melspec(sr=self.hparams['sample_rate']),
            preprocess.pad([[0, 0], [0, 4]], 'CONSTANT', constant_values=self.hparams['log_amin']),
            preprocess.amp_to_log(amin=self.hparams['log_amin']),
        ])

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
        self.generator = create_generator(self.hparams['latent_size'], self.hparams['batch_size'], self.spec_shape)
        self.discriminator = create_discriminator(self.spec_shape)

        # Create the GAN train step
        self.train_step = create_gan_train_step(self.generator,
                                        self.discriminator,
                                        self.generator_loss,
                                        self.discriminator_loss,
                                        self.generator_optimizer,
                                        self.discriminator_optimizer,
                                        self.hparams['batch_size'],
                                        self.hparams['latent_size'])

        # Define some metrics to be used in the training
        self.gen_loss_avg = tf.keras.metrics.Mean()
        self.disc_loss_avg = tf.keras.metrics.Mean()

        self.ckpt = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.generator_optimizer,
            disc_optimizer=self.discriminator_optimizer,
        )

        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(self.save_dir, 'ckpts'), max_to_keep=3)

        self.ckpt.restore(self.manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


        try:
            os.mkdir(os.path.join(self.save_dir, 'images/'))
        except:
            pass

        self.seed = tf.random.normal([self.hparams['num_examples'], self.hparams['latent_size']])

        # Create the training loop
        self.train = create_train_loop(self.dataset,
                                self.train_step,
                                on_epoch_start=self.on_epoch_start,
                                on_step=self.on_step,
                                on_epoch_complete=self.on_epoch_complete)



    # This runs at the start of every epoch
    def on_epoch_start(self, epoch, step):
        self.gen_loss_avg = tf.keras.metrics.Mean()
        self.disc_loss_avg = tf.keras.metrics.Mean()

    # This runs at every step in the training (for each batch in dataset)
    def on_step(self, step, stats):
        gen_loss, disc_loss = stats
        self.gen_loss_avg(gen_loss)
        self.disc_loss_avg(disc_loss)

    # This runs at the end of every epoch and is used to display metrics
    def on_epoch_complete(self, epoch, step):
        self.manager.save()
        display.clear_output(wait=True)
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {self.gen_loss_avg.result()}, Disc Loss: {self.disc_loss_avg.result()}")
        self.generate_and_save_images_epoch(epoch)

    def generate_and_save_images_epoch(self, epoch):
        generated = self.generator(self.seed, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(generated.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated[i, :, :, 0])
            plt.axis('off')

        if self.save_images:
            plt.savefig('{}/images/image_at_epoch_{:04d}.png'.format(self.save_dir, epoch))

        plt.show()
