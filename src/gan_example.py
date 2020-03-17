import tensorflow as tf
import preprocess
import os
from losses import create_simple_gan_loss
from training import Trainer, create_gan_train_step
from datasets.nsynth import nsynth_from_tfrecord, instruments, nsynth_to_melspec
from models.simple_gan import create_generator, create_discriminator
import matplotlib.pyplot as plt
import librosa

import argparse
#import IPython.display as display
from model import Model

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_datasets as tfds

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class GANExample(Model):
    def __init__(self, dataset, hparams):
        super(GANExample, self).__init__(dataset, hparams)

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

        self.save_images = 'save_images' in self.hparams and self.hparams['save_images']
        self.save_audio = 'save_audio' in self.hparams and self.hparams['save_audio']

        if 'save_dir' in self.hparams:
            self.image_save_dir = os.path.join(self.hparams['save_dir'], 'images/')
            self.audio_save_dir = os.path.join(self.hparams['save_dir'], 'audio/')
            try:
                if self.save_images:
                    os.mkdir(self.image_save_dir)
                if self.save_audio:
                    os.mkdir(self.audio_save_dir)
            except:
                pass

        self.seed = tf.random.normal([self.hparams['num_examples'], self.hparams['latent_size']])


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

    def instrument_filter(self, x):
        return tf.reshape(tf.math.equal(x['instrument_family'], instruments[self.hparams['instrument']]), [])

    def preprocess(self, dataset):
        # Determine shape of the spectograms in the dataset
        self.spec_shape = None
        for e in dataset.take(1):
            self.spec_shape = e.shape
            print(f'Spectogram shape: {self.spec_shape}')

        # Make sure we got a shape before continuing
        assert self.spec_shape is not None, "Could not get spectogram shape"

        # Make sure the dimensions of spectogram is divisible by 4.
        # This is because the generator is going to upscale it's state twice with a factor of 2.
        assert self.spec_shape[0] % 4 == 0 and self.spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

        # Create preprocessing pipeline for shuffling and batching
        return preprocess.pipeline([
            preprocess.set_channels(1),
            preprocess.shuffle(self.hparams['buffer_size']),
            preprocess.batch(self.hparams['batch_size']),
            preprocess.prefetch()
        ])(dataset)


    # This runs at the start of every epoch
    def on_epoch_start(self, epoch, step):
        self.gen_loss_avg = tf.keras.metrics.Mean()
        self.disc_loss_avg = tf.keras.metrics.Mean()

    # This runs at every step in the training (for each batch in dataset)
    def on_step(self, step, stats):
        gen_loss, disc_loss = stats
        self.gen_loss_avg(gen_loss)
        self.disc_loss_avg(disc_loss)
        if step % 100 == 0:
            print(f"Step: {step}, Gen Loss: {self.gen_loss_avg.result()}, Disc Loss: {self.disc_loss_avg.result()}")

    # This runs at the end of every epoch and is used to display metrics
    def on_epoch_complete(self, epoch, step, duration):
        #display.clear_output(wait=True)
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {self.gen_loss_avg.result()}, Disc Loss: {self.disc_loss_avg.result()}, Duration: {duration} s")
        self.generate_and_save(epoch, step)

    def generate_and_save(self, epoch, step):
        generated = self.generator(self.seed, training=False)
        generated = tf.reshape(generated, [-1, *self.spec_shape])

        fig = plt.figure(figsize=(4,4))

        for i in range(generated.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated[i, :, :])
            plt.axis('off')

        if self.save_images:
            if self.image_save_dir is None:
                raise Exception("Could not save image, no save_dir was specified in hparams.")
            plt.savefig(os.path.join(self.image_save_dir, 'image_at_epoch_{:04d}_step_{}.png'.format(epoch, step)))

        if self.hparams['plot']:
            plt.show()

        if self.save_audio:
            print("Inverting spectograms...")

            audio = list(preprocess.invert_log_melspec(self.hparams['sample_rate'])(tf.unstack(generated)))
            concat_audio = tf.concat(audio, axis=0)
            print(concat_audio.shape)

            print("Saving audio files...")
            librosa.output.write_wav(os.path.join(self.audio_save_dir, 'audio_at_epoch_{:04d}_step_{}.wav'.format(epoch, step)),
                              tf.reshape(concat_audio, [-1]).numpy(),
                              self.hparams['sample_rate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start training.')
    parser.add_argument('--plot', help='Enable to activate plotting', action='store_true')
    parser.add_argument('--saveimg', help='Enable to save images after each epoch', action='store_true')
    parser.add_argument('--saveaudio', help='Enable to save audio after each epoch', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.plot:
        print('Plotting enabled')
    if args.saveimg:
        print('Saving images enabled')
    if args.saveaudio:
        print('Saving audio enabled')

    # Setup hyperparameters
    hparams = {
        'plot': args.plot,
        'epochs': 100,
        'steps_per_epoch': 100,
        'sample_rate': 16000,
        'batch_size': 32,
        'buffer_size': 1000,
        'latent_size': 100,
        'generator_scale': 128,
        'gen_lr': 0.0001,
        'disc_lr': 0.0004,
        'log_amin': 1e-5,
        'num_examples': 16,
        'save_dir': './',
        'save_images': args.saveimg,
        'save_audio': args.saveaudio,
    }

    # Load nsynth dataset from tfds
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    dataset = nsynth_to_melspec(dataset, hparams)
    gan = GANExample(dataset, hparams)
    gan.train()
