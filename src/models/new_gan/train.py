import tensorflow as tf
import librosa
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load, invert
from models.new_gan.model import GAN


def start(hparams):
    dataset, stats = load(hparams)

    def resize(image, down_scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 1024, 2]), [128//down_scale, 1024//down_scale]))

    dataset = dataset.map(lambda mag, phase, pitch: (tf.stack([mag, phase], axis=-1), pitch))

    gan = GAN(hparams, stats)
    g_init, d_init, gan_init = gan.get_initial_models()

    for i in range(1):#range(hparams['n_blocks']):
        down_scale = 2**(hparams['n_blocks']-i-1)
        batch_size = hparams['batch_sizes'][i]

        scaled_dataset = pro.pipeline([
            pro.map_transform(lambda magphase, pitch: (resize(magphase, down_scale), pitch)),
            pro.cache(),
        ])(dataset)

        gan.train_epochs(g_init, d_init, gan_init, scaled_dataset, 4, batch_size)

    gen = g_init(tf.random.normal([1, hparams['latent_dim']]), training=False)
    plot_magphase(hparams, tf.squeeze(gen), f'generated_magphase')
       

def plot_magphase(hparams, magphase, name, pitch=None):
    mag, phase = tf.unstack(magphase, axis=-1)
    fig, axs = plt.subplots(1, 2)
    if pitch is not None:
        plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
    axs[0].set_title("Magnitude")
    axs[0].imshow(tf.transpose(mag, [1, 0]))
    axs[1].set_title("Phase")
    axs[1].imshow(tf.transpose(phase, [1, 0]))
    plt.savefig(f'{name}.png')

def invert_magphase(hparams, stats, magphase, name):
    magphase = tf.squeeze(magphase)
    mag, phase = tf.unstack(magphase, axis=-1)
    audio = invert(hparams, stats)((mag, phase))
    librosa.output.write_wav(f'{name}.wav', audio.numpy(), sr=hparams['sample_rate'])
