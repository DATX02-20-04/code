import tensorflow as tf
import data.process as preprocess
import os
import matplotlib.pyplot as plt

from models.common.training import Trainer
from data.nsynth import nsynth_from_tfrecord, instruments, nsynth_to_melspec
from models.gan.model import GAN

#import IPython.display as display

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_datasets as tfds

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Define some metrics to be used in the training
gen_loss_avg = tf.keras.metrics.Mean()
disc_loss_avg = tf.keras.metrics.Mean()


def instrument_filter(hparams):
    def _filter(x):
        return tf.reshape(tf.math.equal(x['instrument_family'], instruments[hparams['instrument']]), [])
    return _filter

# This runs at the start of every epoch
def on_epoch_start(epoch, step):
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()

# This runs at every step in the training (for each batch in dataset)
def on_step(epoch, step, stats):
    gen_loss, disc_loss = stats
    gen_loss_avg(gen_loss)
    disc_loss_avg(disc_loss)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}")

# This runs at the end of every epoch and is used to display metrics
def on_epoch_complete(epoch, step, duration):
    #display.clear_output(wait=True)
    print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}, Duration: {duration} s")

def start(hparams):
    # Load nsynth dataset from tfds
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    dataset = nsynth_to_melspec(dataset, hparams)

    # Determine shape of the spectograms in the dataset
    spec_shape = None
    for e in dataset.take(1):
        spec_shape = e.shape
        print(f'Spectogram shape: {spec_shape}')

    # Make sure we got a shape before continuing
    assert spec_shape is not None, "Could not get spectogram shape"

    # Make sure the dimensions of spectogram is divisible by 4.
    # This is because the generator is going to upscale it's state twice with a factor of 2.
    assert spec_shape[0] % 4 == 0 and spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

    # Create preprocessing pipeline for shuffling and batching
    dataset = preprocess.pipeline([
        preprocess.set_channels(1),
        preprocess.shuffle(hparams['buffer_size']),
        preprocess.batch(hparams['batch_size']),
        preprocess.prefetch()
    ])(dataset)

    gan = GAN(spec_shape, hparams)

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(gan.train_step)
    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step
    trainer.on_epoch_complete = on_epoch_complete

    trainer.run()
