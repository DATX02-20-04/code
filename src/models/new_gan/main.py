import tensorflow as tf
import numpy as np
from functools import partial
import librosa
import util
import os
from multiprocessing import Pool
import data.process as pro
import matplotlib.pyplot as plt
from models.new_gan.process import load as gan_load
from models.upscaler.process import load as upscaler_load
from models.new_gan.model import GAN
from models.new_gan.train import plot_magphase
# from models.new_gan.process import invert, invert_griffin
from util import load_hparams
from models.new_gan.process import load as gan_load
from models.upscaler.model import Upscaler

def invert(hparams, stats, upscaler, dataset):
    mag_dataset = pro.pipeline([
        pro.denormalize(normalization='neg_one_to_one', stats=stats),
        pro.log_to_amp(),
        pro.map_transform(lambda x: tf.transpose(x, [1, 0])),
        pro.map_transform(lambda x: tf.py_function(lambda z: librosa.feature.inverse.mel_to_stft(z.numpy(), sr=hparams['sample_rate'], n_fft=hparams['n_fft']), [x], tf.float32)),
        pro.map_transform(lambda x: tf.transpose(x, [1, 0])),
        #pro.map_transform(lambda x: x[:, :-1]),
        ])(dataset)
    dataset = pro.index_map(1, pro.pipeline([
        pro.map_transform(lambda x: tf.reshape(x, [-1, 128, 1025, 1])),
        pro.map_transform(lambda x: x[:, :, :-1, :]),
        pro.map_transform(lambda x: upscaler.model(x, training=False)),
        pro.map_transform(lambda x: tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0]])),
        pro.map_transform(lambda x: tf.squeeze(x)),
        # pro.map_transform(lambda x: x * np.pi),
        # pro.map_transform(lambda x: tf.math.cumsum(x, axis=0)),
        # pro.map_transform(lambda x: (x + np.pi) % (2 * np.pi) - np.pi),
        pro.cast(tf.complex64),
        ]))((mag_dataset, mag_dataset))
    dataset = pro.index_map(0, pro.pipeline([
        pro.cast(tf.complex64),
        ]))(dataset)
    dataset = pro.map_transform(lambda mag, phase: mag * tf.math.exp(1j * phase))(dataset)
    dataset = pro.pipeline([
        pro.map_transform(lambda x: tf.transpose(x, [1, 0])),
        pro.map_transform(lambda x: tf.py_function(lambda z: librosa.core.istft(z.numpy(),
                                                                                #sr=hparams['sample_rate'],
                                                                                win_length=hparams['frame_length'],
                                                                                hop_length=hparams['frame_step'],
                                                                                #n_fft=hparams['n_fft']
                                                                                ),
                                                    [x], tf.float32)),
    ])(dataset)
    return dataset

def invert_griffin(hparams, stats, dataset):
    return pro.pipeline([
        pro.pipeline([
            pro.denormalize(normalization='neg_one_to_one', stats=stats),
            pro.log_to_amp(),
            pro.map_transform(lambda x: tf.transpose(x, [1, 0])),
            pro.map_transform(lambda x: tf.py_function(lambda z: librosa.feature.inverse.mel_to_audio(z.numpy(),
                                                                                            sr=hparams['sample_rate'],
                                                                                            win_length=hparams['frame_length'],
                                                                                            hop_length=hparams['frame_step'],
                                                                                            n_fft=hparams['n_fft']),
                                                       [x], tf.float32)),
        ])
    ])(dataset)

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
    n_producers = 4

    if kwargs['inversion_method'] == 'phase_gen':
        inverter = partial(invert, hparams, gan_stats, upscaler)
    elif kwargs['inversion_method'] == 'griffin':
        inverter = partial(invert_griffin, hparams, gan_stats)

    def run(noise, pitch):
        span('start', 'note_spec_gen')

        pitch = tf.one_hot(pitch, hparams['pitches'])

        n_p = len(pitch)
        batches = n_p // batch_size
        last_batch = int((n_p/batch_size - batches)*batch_size)

        logger(f"pitch={pitch.shape}, batches={batches}, last_batch={last_batch}", level='debug')
        logger(f"noise={noise.shape}", level='debug')

        spectrograms = []
        for i in range(0, batches*batch_size, batch_size):
            spectrogram = generator([noise[i:i+batch_size], pitch[i:i+batch_size]], training=False)
            spectrograms.append(spectrogram)

        if last_batch > 0:
            spectrogram = generator([noise[-last_batch:], pitch[-last_batch:]], training=False)
            spectrograms.append(spectrogram)

        spectrograms = np.concatenate(spectrograms, axis=0)
        spectrograms = list(map(np.squeeze, spectrograms))

        assert len(spectrograms) == len(pitch), "Didn't generate same amount of spectrograms as pitches."

        span('end', 'note_spec_gen')

        n = len(spectrograms)
        span('start', f'{inv_method}_spec_to_wave')
        with Pool(n_producers) as pool:
            notes = pool.map(inverter, spectrograms)
        span('end', f'{inv_method}_spec_to_wave')

        notes = np.concatenate(notes, axis=0)

        assert len(notes) == len(pitch), "Didn't invert same amount of notes as pitches."

        return notes

    return run
