import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load, invert
from models.new_gan.model import GAN
from models.new_gan.train import plot_magphase, invert_magphase


def start(hparams):
    dataset, stats = load(hparams)

    def resize(image, down_scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 1024, 2]), [128//down_scale, 1024//down_scale]))

    # Stack mag and phase into one tensor
    dataset = dataset.map(lambda mag, phase, pitch: (tf.stack([mag, phase], axis=-1), pitch))

    gan = GAN(hparams, stats)

    for i in range(1, hparams['n_blocks']):
        down_scale = 2**(hparams['n_blocks']-i-1)
        batch_size = hparams['batch_sizes'][i]
        epochs = 1

        scaled_dataset = pro.pipeline([
            pro.map_transform(lambda magphase, pitch: (resize(magphase, down_scale), pitch)),
            pro.cache(),
        ])(dataset).take(batch_size)

        [g_normal, g_fadein] = gan.generators[i]
        [d_normal, d_fadein] = gan.discriminators[i]
        [gan_normal, gan_fadein] = gan.models[i]

        print("\nFading in next...")
        gan.train_epochs(g_fadein, d_fadein, gan_fadein, scaled_dataset, epochs, batch_size, True)

        print("\nNormal training...")
        gan.train_epochs(g_normal, d_normal, gan_normal, scaled_dataset, epochs, batch_size)
        print(f"\nBlock {i+1} Size {128//down_scale} {1024//down_scale}")
       
