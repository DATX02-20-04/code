import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from models.new_gan.process import load, invert


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
