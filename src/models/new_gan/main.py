import tensorflow as tf
import numpy as np
import librosa
import util
import os
from multiprocessing import Queue, Lock, Process
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
    n_producers = 4

    def producer(pid, queue, lock, logger, spectrograms):
        with lock:
            logger(f"Producer {pid} started with {len(spectrograms)}.", level='debug')

        for i, spectrogram in spectrograms:
            s = f'pid_{pid}_{inv_method}_spec_to_wave{i+1}'
            with lock:
                logger(f"spectrogram={spectrogram.shape}", level='debug')
                span('start', s)
            if kwargs['inversion_method'] == 'phase_gen':
                note = invert(hparams, gan_stats, upscaler)(spectrogram)
            elif kwargs['inversion_method'] == 'griffin':
                note = invert_griffin(hparams, gan_stats)(spectrogram)
            with lock:
                span('end', s)
            queue.put((i, note))

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
        spectrograms = list(enumerate(map(tf.squeeze, spectrograms)))

        assert len(spectrograms) == len(pitch), "Didn't generate same amount of spectrograms as pitches."

        span('end', 'note_spec_gen')

        queue = Queue()
        lock = Lock()
        producers = []
        specs_per_producer = len(spectrograms) // n_producers

        for i in range(n_producers-1):
            p = Process(target=producer, args=(i, queue, lock, logger, spectrograms[i*specs_per_producer:i*specs_per_producer+specs_per_producer]))
            p.daemon = True
            producers.append(p)

        p = Process(target=producer, args=(i, queue, lock, logger, spectrograms[(n_producers-1)*specs_per_producer:]))
        p.daemon = True
        producers.append(p)

        for p in producers:
            p.start()

        n = len(spectrograms)
        notes = []
        span('start', f'{inv_method}_spec_to_wave')
        while len(notes) < len(spectrograms):
            spectrogram = tf.squeeze(spectrogram)
            note = queue.get()
            notes.append(note)
        span('end', f'{inv_method}_spec_to_wave')

        for p in producers:
            p.join()

        notes = list(map(lambda x: x[1], sorted(notes)))
        notes = np.concatenate(notes, axis=0)

        assert len(notes) == len(pitch), "Didn't invert same amount of notes as pitches."

        return notes

    return run