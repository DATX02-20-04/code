import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load, invert
from models.new_gan.model import GAN


def start(hparams):
    dataset, stats = load(hparams)

    def resize(image, down_scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 1024, 2]), [128//down_scale, 1024//down_scale]))

    # Stack mag and phase into one tensor
    dataset = dataset.map(lambda mag, phase, pitch: (tf.stack([mag, phase], axis=-1), pitch))


    gan = GAN(hparams, stats)

    print("\nGrowing complete, starting normal training...")
    [g_normal, g_fadein] = gan.generators[-1]
    [d_normal, d_fadein] = gan.discriminators[-1]
    [gan_normal, gan_fadein] = gan.models[-1]
    scaled_dataset = pro.pipeline([
        pro.cache(),
    ])(dataset)
    batch_size = hparams['finished_batch_size']
    gan.train_epochs(g_normal, d_normal, gan_normal, scaled_dataset, 1, batch_size)
       

