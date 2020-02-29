import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import layers
import librosa
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from scipy.io import wavfile
from nsynth import NSynthDataset
import model
import threading

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

buffer_size = 10_000
batch_size = 32
timesteps = 128
channels = 1
latent_size = 100
frame_length = 4048
hop_length = 2196
n_fft = 4048
samples = 64000
n_mels = 28

epochs = 100000

def instrument_filter(x):
    return tf.reshape(tf.equal(x["instrument_family"], 0), [])

dataset = (NSynthDataset('/home/big/datasets/nsynth/nsynth-train.tfrecord')
           .dataset()
           .filter(instrument_filter)
           .extract(['audio'# , 'instrument_family'
           ])
           # .one_hot(1, 11)
           .stft(0, frame_length, hop_length, n_fft)
           .abs(0)
           .melspec(0, n_mels)
           # .logspec(0)
           .flip_axes(0)
           # .unbatch()
           .set_chan(0)
           # .set_chan(1)
           .prefetch())

fig, axs = plt.subplots(3)
plt.ion()
plt.show()

for x, in dataset.examples.take(1):
    shape = x.shape[0:2]

dataset = (dataset
           .shuffle(buffer_size)
           .batch(batch_size)
           .prefetch())

print(shape)

generator = model.create_generator(latent_size, batch_size, shape)
discriminator = model.create_discriminator(shape)

num_examples = 4

seed = tf.random.normal([num_examples, latent_size])

def concat_images(images):
    return tf.concat([tf.reshape(image, shape) for image in tf.unstack(images, axis=0)], axis=1)

def save_and_plot(epoch, save=False, batch=None):
    if save:
        print("Saving audio sample...")

    image = concat_images(generator(seed, training=False))

    if save:
        print("Inverting melspecs...")
        # image = tf.math.exp(image)
        waveform = librosa.feature.inverse.mel_to_audio(
            image.numpy(),
            sr=dataset.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=frame_length)

        print("Inverting done.")



    if batch:
        axs[1].cla()
        axs[1].imshow(concat_images(batch[0][0:num_examples]).numpy())

    axs[0].cla()
    axs[0].imshow(image.numpy())

    if save:
        axs[2].cla()
        print("Writing wav file...")
        wavfile.write('audio/audio_{}_E{:04d}.wav'.format(int(time.time()*1000), epoch), dataset.sr, waveform)
        axs[2].plot(waveform)
        print("Writing wav file done.")

    plt.draw()
    plt.pause(0.1)

    if save:
        print("Save done.")


save_and_plot(0)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0004)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.zeros_like(real_output)+tf.random.uniform(real_output.shape, 0, 0.1),
        real_output)
    fake_loss = cross_entropy(
        tf.ones_like(fake_output)-tf.random.uniform(fake_output.shape, 0, 0.1),
        fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output)


@tf.function
def train_step(model, x):
    noise = tf.random.normal([batch_size, latent_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_x = generator(noise, training=True)
      print(generated_x.shape)

      real_output = discriminator(x, training=True)
      fake_output = discriminator(generated_x, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss



def train(epochs):
    for epoch in range(epochs):
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()
        start = time.time()

        step = 0
        steps = 4000
        for x in dataset.examples.take(steps):
            gen_loss, disc_loss = train_step(model, x)
            gen_loss_avg(gen_loss)
            disc_loss_avg(disc_loss)
            step += 1
            if step % 10 == 0:
                print (f'Loss gen: {gen_loss_avg.result()} loss disc: {disc_loss_avg.result()} {step}')
                if step != steps:
                    if step % 1000 == 0:
                        save_and_plot(epoch, True)
                    elif step % 20 == 0:
                        save_and_plot(epoch, False, x)

        print (f'Loss gen: {gen_loss_avg.result()} loss disc: {disc_loss_avg.result()}, Time for epoch {epoch+1} is {time.time()-start} sec')

        save_and_plot(epoch, True)
        # elif epoch % 2 == 0:

train(epochs)

