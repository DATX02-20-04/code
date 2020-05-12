import tensorflow as tf
import numpy as np
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load, invert
from models.new_gan.model import GAN
from models.new_gan.train import plot_magphase, invert_magphase


def start(hparams):
    dataset, stats = load(hparams)

    def resize(image, scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 32, 256, 2]), [32//scale, 256//scale]))

    scale = 8

    dataset = dataset.map(lambda magphase, pitch: (resize(magphase, scale), pitch))
    dataset = dataset.map(lambda magphase, pitch: (tf.unstack(magphase, axis=-1), pitch))
    dataset = list(dataset.as_numpy_iterator())
    dataset = sorted(dataset, key=lambda x: np.argmax(x[1]))
    mags = map(lambda x: x[0], dataset)
    mags = list(mags)
    mag = [m[0] for m in mags]
    phase = [m[1] for m in mags]
    mags = np.concatenate(mag, axis=0)
    phase = np.concatenate(phase, axis=0)

    plt.figure(figsize=(32, 32))
    plt.title("Magnitude")
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(tf.transpose(mags, [1, 0]), origin='bottom')
    axs[1].imshow(tf.transpose(phase, [1, 0]), origin='bottom')
    plt.tight_layout()
    plt.savefig('increasing_mags.png', bbox_inches='tight')
