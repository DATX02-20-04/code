import tensorflow as tf
import json
import numpy as np
import functools
import data.process as pro
from models.common.training import Trainer
from data.nsynth import nsynth_from_tfrecord, nsynth_to_melspec
from models.gan.model import GAN
import data.process as pro
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io

# Define some metrics to be used in the training
gen_loss_avg = tf.keras.metrics.Mean()
disc_loss_avg = tf.keras.metrics.Mean()


# This runs at the start of every epoch
def on_epoch_start(epoch, step, tsw):
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()


# This runs at every step in the training (for each batch in dataset)
def on_step(epoch, step, stats, tsw):
    gen_loss, disc_loss = stats
    gen_loss_avg(gen_loss)
    disc_loss_avg(disc_loss)
    with tsw.as_default():
        tf.summary.scalar('gen_loss', gen_loss_avg.result(), step=step)
        tf.summary.scalar('disc_loss', disc_loss_avg.result(), step=step)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}")

# This runs at the end of every epoch and is used to display metrics
def on_epoch_complete(epoch, step, duration, tsw, gan):
    #display.clear_output(wait=True)
    count = 5
    seed = tf.random.normal((count, gan.hparams['latent_size']))
    mid = gan.hparams['cond_vector_size']//2
    pitches = tf.one_hot(range(mid-count//2, mid+count//2), gan.hparams['cond_vector_size'], axis=1)

    samples = tf.reshape(gan.generator([seed, pitches], training=False), [-1, 128, 128])
    img = tf.unstack(samples)
    img = tf.reverse(tf.concat(img, axis=1), axis=[0])

    with tsw.as_default():
        tf.summary.image(f'Spectrogram at epoch {epoch}', img, step=step)
    print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}, Duration: {duration} s")

def start(hparams):

    # Load nsynth dataset from tfds
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    gan_stats = calculate_dataset_stats(hparams, dataset)

    dataset = nsynth_to_melspec(dataset, hparams, gan_stats)

    # Determine shape of the spectograms in the dataset
    spec_shape = None
    for x in dataset.take(1):
        e = x['audio']
        cond = x['pitch']
        spec_shape = e.shape
        print(cond)
        print(f'Spectogram shape: {spec_shape}')

    # Make sure we got a shape before continuing
    assert spec_shape is not None, "Could not get spectogram shape"

    # Make sure the dimensions of spectogram is divisible by 4.
    # This is because the generator is going to upscale it's state twice with a factor of 2.
    assert spec_shape[0] % 4 == 0 and spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

    dataset = pro.index_map('audio', pro.reshape([*spec_shape, 1]))(dataset)

    # Create preprocessing pipeline for shuffling and batching
    dataset = pro.pipeline([
        pro.cache(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(dataset)

    gan = GAN(spec_shape, hparams)
    gan.discriminator.summary()
    gan.generator.summary()


    # This runs at the end of every epoch and is used to display metrics
    def on_epoch_complete(epoch, step, duration, tsw):
        #display.clear_output(wait=True)
        count = 6
        seed = tf.random.normal((count, gan.hparams['latent_size']))
        mid = gan.hparams['cond_vector_size']//2
        pitches = tf.one_hot(range(mid-count//2, mid+count//2), gan.hparams['cond_vector_size'], axis=1)

        samples = tf.reshape(gan.generator([seed, pitches], training=False), [-1, 128, 128])
        img = tf.unstack(samples)
        img = tf.reverse(tf.concat(img, axis=1), axis=[0])
        plt.axis('off')
        plt.imshow(img)

        buf = io.BytesIO()
        plt.savefig(buf,  format='png')
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with tsw.as_default():
            tf.summary.image(f'Spectrogram at epoch {epoch}', image, step=step)
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}, Duration: {duration} s")


    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)
    trainer.init_tensorboard()
    trainer.set_train_step(gan.train_step)
    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step
    trainer.on_epoch_complete = on_epoch_complete

    trainer.run()

def calculate_dataset_stats(hparams, dataset):
    print("Calculating dataset stats...")
    dataset = nsynth_to_melspec(dataset, hparams)

    megabatch = dataset.batch(100000).as_numpy_iterator()
    x = next(megabatch)
    x = x['audio']
    mean = x.mean(axis=0)
    min_ = x.min(axis=0)
    max_ = x.max(axis=0)
    variance = x.var(axis=0)

    np.savez('gan_stats.npz', mean=mean, min=min_, max=max_, variance=variance)

    print("Calculating dataset stats, done.")

    return {
        'mean': mean,
        'min': min_,
        'max': max_,
        'variance': variance,
    }
