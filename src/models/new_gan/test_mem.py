import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.model import GAN
from models.new_gan.process import load, invert


def start(hparams):
    dataset, stats = load(hparams)

    def size(hparams, down_scale):
        return [hparams['width']//down_scale, hparams['height']//down_scale]

    def resize(image, down_scale):
        return tf.reshape(tf.image.resize(tf.reshape(image,
                                                     [1, hparams['width'], hparams['height'], 1]),
                                          size(hparams, down_scale)), [*size(hparams, down_scale), 1])

    last = hparams['n_blocks']-1
    dataset = pro.pipeline([
        pro.map_transform(lambda magphase, pitch: (resize(magphase, 1), pitch)),
        pro.cache(),
    ])(dataset)

    init_size = size(hparams, 2**(last))
    print(f"Init size: {init_size}")

    gan = GAN(hparams, stats, init_size)

    [g_normal, g_fadein] = gan.generators[last]
    [d_normal, d_fadein] = gan.discriminators[last]
    [gan_normal, gan_fadein] = gan.models[last]

    gan.train_epochs(g_normal, d_normal, gan_normal, dataset, hparams['epochs'][last], hparams['batch_sizes'][last])

