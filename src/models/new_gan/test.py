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
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 512, 256, 1]), [512//down_scale, 256//down_scale]))

    scale = 1

    for mag, phase, pitch in dataset.skip(9).take(1):
        print(mag.shape, phase.shape)
        mag = resize(mag, scale)
        phase = resize(phase, scale)
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
        axs[0].set_title("Magnitude")
        axs[0].imshow(mag)
        axs[1].set_title("Phase")
        axs[1].imshow(phase)
        plt.savefig('non_inverted_plot_cqt.png')
        audio = invert(hparams, stats)((mag, phase))
        librosa.output.write_wav('inverted_audio.wav', audio.numpy(), sr=hparams['sample_rate'])
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 1024, 2]), [128//down_scale, 1024//down_scale]))

