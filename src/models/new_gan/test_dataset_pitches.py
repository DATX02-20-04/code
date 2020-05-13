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
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 256, 1]), [128//scale, 256//scale]))

    scale = 1

    dataset = dataset.map(lambda mag, pitch: (resize(mag, scale), pitch))
    dataset = list(dataset.as_numpy_iterator())
    dataset = sorted(dataset, key=lambda x: np.argmax(x[1]))
    mags = map(lambda x: x[0], dataset)
    mags = list(mags)
    mags = np.concatenate(mags, axis=0)

    plt.figure(figsize=(16, 16))
    plt.title("Magnitude")
    plt.imshow(tf.transpose(mags, [1, 0]), origin='bottom')
    plt.tight_layout()
    plt.savefig('increasing_mags.png', bbox_inches='tight')
