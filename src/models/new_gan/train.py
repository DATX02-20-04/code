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
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 32, 256, 1]), [32//down_scale, 256//down_scale]))

    gan = GAN(hparams, stats)
    block = tf.Variable(1)

    ckpt = tf.train.Checkpoint(
        gan=gan,
        optimizer=gan.optimizer,
        block=block,
    )

    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(hparams['save_dir'], 'ckpts', hparams['name']),
                                         max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {} block {}".format(manager.latest_checkpoint, block.numpy()))
    else:
        print("Initializing from scratch.")

    if block == 1:
        # Get the first models to train
        g_init, d_init, gan_init = gan.get_initial_models()

        # Create the smallest scaled dataset to train the first
        scaled_dataset = pro.pipeline([
            pro.map_transform(lambda magphase, pitch: (resize(magphase, 2**(hparams['n_blocks']-1)), pitch)),
            pro.cache(),
        ])(dataset)

        gan.train_epochs(g_init, d_init, gan_init, scaled_dataset, hparams['epochs'][0], hparams['batch_sizes'][0])
        gen = g_init(tf.random.normal([5, hparams['latent_dim']]), training=False)
        plot_magphase(hparams, gen, f'generated_magphase_block00')

    for i in range(block.numpy(), hparams['n_blocks']):
        down_scale = 2**(hparams['n_blocks']-i-1)
        batch_size = hparams['batch_sizes'][i]
        epochs = hparams['epochs'][i]

        scaled_dataset = pro.pipeline([
            pro.map_transform(lambda magphase, pitch: (resize(magphase, down_scale), pitch)),
            pro.cache(),
        ])(dataset)

        [g_normal, g_fadein] = gan.generators[i]
        [d_normal, d_fadein] = gan.discriminators[i]
        [gan_normal, gan_fadein] = gan.models[i]

        print("\nFading in next...")
        gan.train_epochs(g_fadein, d_fadein, gan_fadein, scaled_dataset, epochs, batch_size, True)

        print("\nNormal training...")
        gan.train_epochs(g_normal, d_normal, gan_normal, scaled_dataset, epochs, batch_size)

        block.assign_add(1)
        manager.save()

        gen = g_normal(tf.random.normal([5, hparams['latent_dim']]), training=False)
        plot_magphase(hparams, gen, f'generated_magphase_block{i:02d}')

    final_epochs = 100
    batch_size = 16
    last = hparams['n_blocks']-1
    [g_normal, g_fadein] = gan.generators[last]
    [d_normal, d_fadein] = gan.discriminators[last]
    [gan_normal, gan_fadein] = gan.models[last]
    for i in range(1, final_epochs+1):
        print(f"\nFinal training {i}/{final_epochs}...")
        gan.train_epochs(g_normal, d_normal, gan_normal, dataset, 1, batch_size)

        manager.save()
       

def plot_magphase(hparams, magphase, name, pitch=None):
    assert len(magphase.shape) == 4, "Magphase needs to be in the form (batch, width, height, channels)"
    count = magphase.shape[0]
    fig, axs = plt.subplots(1, 2*count)
    for i in range(count):
        mag = tf.squeeze(magphase[i])
        if pitch is not None:
            plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
        axs[0+i*2].set_title("Mag")
        axs[0+i*2].axes.get_xaxis().set_visible(False)
        axs[0+i*2].axes.get_yaxis().set_visible(False)
        axs[0+i*2].imshow(tf.transpose(mag, [1, 0]))

    plt.tight_layout()
    plt.savefig(f'{name}.png')

