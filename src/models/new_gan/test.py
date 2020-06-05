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
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 256, 1]), [128//scale, 256//scale]))

    scale = 1

    for mag, pitch in dataset:
        p = tf.argmax(pitch)
        if p == 14:
            print(mag.shape, )
            mag = resize(mag, scale)
            fig, axs = plt.subplots(1, 1)
            plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
            axs.set_title("Magnitude")
            axs.imshow(tf.transpose(mag, [1, 0]))
            plt.savefig('non_inverted_plot.png')
            audio = invert(hparams, stats)(mag)
            librosa.output.write_wav('inverted_audio.wav', audio.numpy(), sr=hparams['sample_rate'])
            print("PITCH FOUND")
            exit()
