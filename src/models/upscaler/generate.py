import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.upscaler.process import load, invert
from models.upscaler.train import plot_magphase, invert_magphase
from models.upscaler.model import Upscaler


def start(hparams):
    dataset, stats = load(hparams)

    upscaler = Upscaler(hparams, stats)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    upscaler.model.load_weights(checkpoint_path)

    for x, y in dataset.skip(4).take(1):
        print(x.shape, y.shape)
        upscaled = upscaler.model(tf.reshape(x, [1, 32, 256, 1]), training=False)
        upscaled = tf.squeeze(upscaled)

        u_m, u_p = tf.unstack(upscaled, axis=-1)
        y_m, y_p = tf.unstack(y, axis=-1)
        fig, axs = plt.subplots(1, 5)

        axs[0].set_title("Inp Mag")
        axs[0].imshow(tf.transpose(x, [1, 0]))

        axs[1].set_title("Tar Mag")
        axs[1].imshow(tf.transpose(y_m, [1, 0]))
        axs[2].set_title("Tar Phase")
        axs[2].imshow(tf.transpose(y_p, [1, 0]))

        axs[3].set_title("Gen Mag")
        axs[3].imshow(tf.transpose(u_m, [1, 0]))
        axs[4].set_title("Gen Pha")
        axs[4].imshow(tf.transpose(u_p, [1, 0]))

        plt.savefig('generated_plot.png')

        audio = invert(hparams, stats)((u_m, u_p))
        librosa.output.write_wav('inverted_audio.wav', audio.numpy(), sr=hparams['sample_rate'])
        audio = invert(hparams, stats)((y_m, y_p))
        librosa.output.write_wav('inverted_audio_real.wav', audio.numpy(), sr=hparams['sample_rate'])
