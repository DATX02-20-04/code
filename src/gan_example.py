import tensorflow as tf
import preprocess
from losses import create_simple_gan_loss
from training import create_train_loop, create_gan_train_step
from datasets.nsynth import nsynth_from_tfrecord
from models.simple_gan import create_generator, create_discriminator

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Setup hyperparameters
hparams = {
    'epochs': 10,
    'steps_per_epoch': 10,
    'sample_rate': 16000,
    'batch_size': 64,
    'buffer_size': 1000,
    'latent_size': 100,
    'gen_lr': 0.0001,
    'disc_lr': 0.0004
}

# Load nsynth dataset from a tfrecord
dataset = nsynth_from_tfrecord('/home/big/datasets/nsynth/nsynth-train.tfrecord')

# Create preprocessing pipeline for the melspectograms
dataset = preprocess.pipeline(dataset, [
    preprocess.extract('audio'),
    preprocess.melspec(sr=hparams['sample_rate']),
    preprocess.amp_to_log(),
    preprocess.pad([[0, 0], [6, 0]], 'CONSTANT')
])

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
dataset = preprocess.pipeline(dataset, [
    preprocess.set_channels(1),
    preprocess.shuffle(hparams['buffer_size']),
    preprocess.batch(hparams['batch_size']),
    preprocess.prefetch()
])

# Create the generator and discriminator loss functions
generator_loss, discriminator_loss = create_simple_gan_loss(tf.keras.losses.BinaryCrossentropy(from_logits=True))

# Define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(hparams['gen_lr'])
discriminator_optimizer = tf.keras.optimizers.Adam(hparams['disc_lr'])

# Create the actual models for the generator and discriminator
generator = create_generator(hparams['latent_size'], hparams['batch_size'], spec_shape)
discriminator = create_discriminator(spec_shape)

# Create the GAN train step
train_step = create_gan_train_step(generator,
                                   discriminator,
                                   generator_loss,
                                   discriminator_loss,
                                   generator_optimizer,
                                   discriminator_optimizer,
                                   hparams['batch_size'],
                                   hparams['latent_size'])

# Define some metrics to be used in the training
gen_loss_avg = tf.keras.metrics.Mean()
disc_loss_avg = tf.keras.metrics.Mean()

# This runs at the start of every epoch
def on_epoch_start(epoch, step):
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()

# This runs at every step in the training (for each batch in dataset)
def on_step(step, stats):
    gen_loss, disc_loss = stats
    gen_loss_avg(gen_loss)
    disc_loss_avg(disc_loss)

# This runs at the end of every epoch and is used to display metrics
def on_epoch_complete(epoch, step):
    print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}")

# Create the training loop
train = create_train_loop(dataset,
                          train_step,
                          on_epoch_start=on_epoch_start,
                          on_step=on_step,
                          on_epoch_complete=on_epoch_complete)

# Start training
train(epochs=hparams['epochs'], steps=hparams['steps_per_epoch'])
