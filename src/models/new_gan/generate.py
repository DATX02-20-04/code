import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load as gan_load
from models.new_gan.model import GAN
from models.new_gan.train import plot_magphase
from models.new_gan.process import invert
from util import load_hparams

def start(hparams):
    dataset, stats = gan_load(hparams)

    gan = GAN(hparams, stats)
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

    _, upscaler_stats = upscaler_load(upscaler_hparams)
    upscaler = Upscaler(upscaler_hparams, upscaler_stats)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    try:
        upscaler.model.load_weights(checkpoint_path)
    except:
        print("Initializing from scratch.")

    [g_normal, g_fadein] = gan.generators[-1]

    gen = g_normal([seed, seed_pitches], training=False)
    print("GEN SHAPE", gen.shape)

    gen = tf.concat(tf.unstack(gen), axis=0)

    pitch = 0
    x_m = tf.squeeze(gen)
    fig, axs = plt.subplots(1, 1)

    axs.set_title("Gen Mag")
    axs.imshow(tf.transpose(x_m, [1, 0]))

    plt.savefig(f'generated_final_plot{pitch}.png')

    audios = []
    for sample in samples:
        print(sample.shape)
        audio = invert(gan_hparams, gan_stats, upscaler)(sample)
        audios.append(audio)
    librosa.output.write_wav(f'inverted_final_audio{pitch}.wav', audio.numpy(), sr=hparams['sample_rate'], norm=True)
       

