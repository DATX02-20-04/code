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

    def size(down_scale):
        return [hparams['height']//down_scale, hparams['width']//down_scale]

    def resize(image, down_scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image,
            [1, hparams['height'], hparams['width'], 1]),
            size(down_scale)))

    scale = 1

    for i, (mag, phase, pitch) in enumerate(dataset.take(4)):
        print(mag.shape, phase.shape)
        mag = resize(mag, scale)
        phase = resize(phase, scale)
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
        axs[0].set_title("Magnitude")
        axs[0].imshow(mag)
        axs[1].set_title("Phase")
        axs[1].imshow(phase)
        plt.savefig(f'non_inverted_plot_cqt{i}.png')
        audio = invert(hparams, stats)((mag, phase))
        librosa.output.write_wav(f'inverted_audio{i}.wav', audio.numpy(), sr=hparams['sample_rate'], norm=True)

