import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.model import GAN
from models.new_gan.process import load, invert


def start(hparams):
    dataset, stats = load(hparams)

    # Stack mag and phase into one tensor
    dataset = dataset.map(lambda mag, phase, pitch: (tf.stack([mag, phase], axis=-1), pitch))


    gan = GAN(hparams, stats)

    last = hparams['n_blocks']-1

    [g_normal, g_fadein] = gan.generators[last]
    [d_normal, d_fadein] = gan.discriminators[last]
    [gan_normal, gan_fadein] = gan.models[last]

    gan.train_epochs(g_normal, d_normal, gan_normal, dataset, hparams['epochs'][last], hparams['batch_sizes'][last])

