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

    def resize(image, scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 1024, 1]), [128//scale, 1024//scale]))

    scale = 1

    for mag, phase, pitch in dataset.skip(4).take(1):
        print(mag.shape, phase.shape)
        mag = resize(mag, scale)
        phase = resize(phase, scale)
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
        axs[0].set_title("Magnitude")
        axs[0].imshow(tf.transpose(mag, [1, 0]))
        axs[1].set_title("Phase")
        axs[1].imshow(tf.transpose(phase, [1, 0]))
        plt.savefig('non_inverted_plot.png')
        audio = invert(hparams, stats)((mag, phase))
        librosa.output.write_wav('inverted_audio.wav', audio.numpy(), sr=hparams['sample_rate'])
