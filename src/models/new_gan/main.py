import tensorflow as tf
import librosa
import util
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load as gan_load
from models.upscaler.process import load as upscaler_load
from models.new_gan.model import GAN
from models.new_gan.train import plot_magphase
from models.new_gan.process import invert, invert_griffin
from util import load_hparams
from models.new_gan.process import load as gan_load
from models.upscaler.model import Upscaler

def create_run(hparams, logger, span, **kwargs):
    upscaler_hparams = util.load_hparams('hparams/upscaler.yml')
    _, gan_stats = gan_load(hparams)
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
        logger("Restored from {} block {}".format(manager.latest_checkpoint, block.numpy()))
    else:
        logger("Initializing from scratch.")

    _, upscaler_stats = upscaler_load(upscaler_hparams)
    upscaler = Upscaler(upscaler_hparams, upscaler_stats)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    try:
        upscaler.model.load_weights(checkpoint_path)
    except:
        logger("Initializing from scratch.")

    [generator, _] = gan.generators[-1]

    batch_size = kwargs['batch_size']
    inv_method = kwargs['inversion_method']

    def run(noise, pitch):
        span('start', 'note_spec_gen')
        spectrograms = generator([seed, pitch], training=False)
        span('end', 'note_spec_gen')

        spectrograms = tf.unstack(tf.reshape(spectrograms, [-1, 128, 256]))
        n = len(spectrograms)
        notes = []
        span('start', f'{inv_method}_spec_to_wave')
        for i, spectrogram in enumerate(spectrograms):
            s = f'{inv_method}_spec_to_wave{i+1}/{n}'
            span('start', s)
            if kwargs['inversion_method'] == 'phase_gen':
                note = invert(hparams, gan_stats, upscaler)(spectrogram)
            elif kwargs['inversion_method'] == 'griffin':
                note = invert_griffin(hparams, gan_stats)(spectrogram)
            span('end', s)
            notes.append(note)
        span('end', 'spec_to_wave')

        return tf.concat(notes, axis=0)
