import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import wavfile
from nsynth import NSynthDataset
from model import create_model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

buffer_size = 1024
batch_size = 64
timesteps = 128
channels = 1

epochs = 100

def instrument_filter(x):
    return tf.reshape(tf.equal(x["instrument_family"], 0), [])

dataset = (NSynthDataset('/home/big/datasets/nsynth/nsynth-train.tfrecord')
           .dataset()
           .filter(instrument_filter)
           .audio()
           .audio_padstart(timesteps)
           .audio_windows(timesteps+1)
           .audio_split([timesteps, 1])
           # .audio_stft()
           # .stft_magspec()
           # .magspec_melspec()
           .shuffle(buffer_size)
           .batch(batch_size)
           .prefetch())

plt.ion()
plt.show()


model = create_model(timesteps, channels)
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)

        batch_loss = loss(y_pred, y)

    gradients = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return batch_loss


def generate_waveform(model, samples):
    history = tf.unstack(tf.random.normal([1, timesteps, channels]), axis=1)
    waveform = []

    for i in range(samples):
        histtens = tf.reshape(tf.concat(history, 0), [1, timesteps, channels])
        y_pred = model(histtens, training=False)
        del history[0]
        history.append(y_pred)
        waveform.append(y_pred)
        if i % 1000 == 0:
            print(f"Waveform sample {i}/{samples}")

    return tf.concat(waveform, 0)


def save_and_plot(epoch):
    waveform = generate_waveform(model, 64000).numpy().reshape(-1)
    wavfile.write('audio/audio_at_epoch_{:04d}.wav'.format(epoch), dataset.sr, waveform)

    plt.clf()
    plt.plot(waveform)
    plt.draw()
    plt.pause(0.1)


for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    start = time.time()

    step = 0
    steps = 10000
    for x, y in dataset.examples.take(steps):
        batch_loss = train_step(model, x, y)
        epoch_loss_avg(batch_loss)
        step += 1
        if step % 1000 == 0:
            print (f'Loss: {epoch_loss_avg.result()} {int((step/steps)*100)}%')

    print (f'Loss: {epoch_loss_avg.result()}, Time for epoch {epoch+1} is {time.time()-start} sec')

    if (epoch+1) % 10 == 0:
        save_and_plot(epoch)


