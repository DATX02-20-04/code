import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load as gan_load
from models.upscaler.process import load as upscale_load
from models.new_gan.model import GAN
from models.new_gan.train import plot_magphase
from models.upscaler.model import Upscaler
from models.upscaler.process import invert as upscale_invert
from util import load_hparams

def start(hparams):
    dataset, gan_stats = gan_load(hparams)
    dataset, upscale_stats = upscale_load(hparams)

    gan = GAN(hparams, gan_stats)
    block = tf.Variable(0)
    step = tf.Variable(0)
    pitch_start = 0
    pitch_end = hparams['pitches']
    step_size = 1
    seed_pitches = tf.range(pitch_start, pitch_start+pitch_end, step_size)
    seed = tf.Variable(tf.random.normal([seed_pitches.shape[0], hparams['latent_dim']]))
    seed_pitches = tf.one_hot(seed_pitches, hparams['pitches'], axis=1)

    ckpt = tf.train.Checkpoint(
        gan=gan,
        seed=seed,
        generator_optimizer=gan.generator_optimizer,
        discriminator_optimizer=gan.discriminator_optimizer,
        block=block,
        step=step,
    )

    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(hparams['save_dir'], 'ckpts', hparams['name']),
                                         max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {} block {}".format(manager.latest_checkpoint, block.numpy()))
    else:
        print("Initializing from scratch.")


    [g_normal, g_fadein] = gan.generators[hparams['n_blocks']-1]

    gen = g_normal([seed, seed_pitches], training=False)
    print("GEN SHAPE", gen.shape)
    plot_magphase(hparams, gen, f'generated_magphase_block_last')

    upscale_hparams = load_hparams('hparams/upscaler.yml')
    upscaler = Upscaler(upscale_hparams, upscale_stats)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    upscaler.model.load_weights(checkpoint_path)

    gens = tf.split(gen, num_or_size_splits=[30, 30, 1])
    gens = gens[:-1]
    upscaleds = [upscaler.model(gen, training=False) for gen in gens]

    pitch = 0
    for upscaled in upscaleds:
        for i in range(upscaled.shape[0]):
            u_m, u_p = tf.unstack(upscaled[i], axis=-1)
            print(pitch)
            x_m = tf.squeeze(gen[i])
            fig, axs = plt.subplots(1, 4)

            axs[0].set_title("Gen Mag")
            axs[0].imshow(tf.transpose(x_m, [1, 0]))

            axs[2].set_title("Ups Mag")
            axs[2].imshow(tf.transpose(u_m, [1, 0]))
            axs[3].set_title("Ups Pha")
            axs[3].imshow(tf.transpose(u_p, [1, 0]))

            plt.savefig(f'generated_upscale_plot{pitch}.png')

            audio = upscale_invert(upscale_hparams, upscale_stats)((u_m, u_p))
            librosa.output.write_wav(f'inverted_upscale_audio{pitch}.wav', audio.numpy(), sr=upscale_hparams['sample_rate'], norm=True)
            pitch += 1
           

