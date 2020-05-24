import util
import tensorflow as tf
import importlib
import numpy as np
import librosa
import random
import os
import time
import scripts.gen_tone
import models.transformer.generate as transformer
import tensorflow_datasets as tfds
from models.new_gan.process import load as gan_load
from models.upscaler.process import load as upscaler_load
from models.common.training import Trainer
from models.upscaler.model import Upscaler
from models.new_gan.process import invert
import data.process as pro
import data.midi as M

def create_get_prior(hparams, melody):
    dataset = tf.data.Dataset.list_files(f"{hparams['dataset_root']}/**/*.midi")

    dataset_single = pro.pipeline([
        pro.midi(),
        pro.frame(hparams['frame_size'], hparams['frame_hop_len'], True),
        pro.unbatch(),
    ])(dataset).repeat()

    def get_prior():
        return next(dataset_single.skip(random.randint(0, melody['max_prior'])).as_numpy_iterator())

    return get_prior

def render(notes, times, tone_len):
    # Create output tensor for the waveform
    waveform = tf.zeros(max(times) + tone_len)

    # Add each note to the output waveform
    for time, note in zip(times, notes):
        note = note[:sr]
        waveform += tf.pad(sound, [(time, len(out)-len(sound)-time)])

    return waveform

def start(hparams):
    log_level   = hparams['log_level']
    models   = hparams['models']
    hmelody   = hparams['melody']
    hnote     = hparams['note']
    sr       = hparams['sample_rate']
    spn      = hnote['samples_per_note']
    tone_len = hnote['tone_length']

    # Model run function
    mrun = {}

    # Model hyperparameters
    mhparams = {}

    # Load model hyperparameters
    for part, model in models.items():
        hp = util.load_hparams(f'hparams/{model}.yml')
        mhparams[part] = hp

        main = importlib.import_module(f'models.{model}.main')
        logger = util.create_logger(part, log_level=log_level)
        span = util.create_span(logger)
        mrun[part] = main.create_run(hp, logger, span, **hparams[part])

    # Create prior function
    get_prior = create_get_prior(mhparams['melody'], hmelody)

    logger = util.create_logger('renderer', log_level=log_level)

    # Record starting time of rendering
    span = util.create_span(logger)
    start_time = span('start', 'render_system')

    if hmelody['generate']:
        # Ger random prior
        prior = get_prior()

        # Generate the melody to render given the prior
        melody = mrun['melody'](prior)

        # Decode midi tokens to midi
        midi = pro.decode_midi()(melody)
    else:
        logger(f"Loading melody from midi file: {melody['midi_file']}")
        prior = None
        with open(hmelody['midi_file'], 'rb') as f:
            midi = M.read_midi(f)

    # Flatten midi to a list of events
    midi = midi.flatten()

    # Create tensors of the pitch, amplitude and velocity in the format for the note generator
    pitches = tf.cast([a.pitch - hmelody['note_offset'] for a in midi if isinstance(a, M.Midi.NoteEvent)], tf.int32)
    amps     = tf.cast([a.velocity / 127 for a in midi if isinstance(a, M.Midi.NoteEvent)], tf.float32)
    vels     = tf.ones_like(pitches) * hmelody['velocity']

    # Create the noise for the note generator
    noise = tf.random.normal((len(pitches), mhparams['note']['latent_dim']))

    # Generate the notes
    notes = mrun['note'](noise, pitches) * amps

    # Calculate the times each note should be placed
    times = [int(a.time * spn) for a in midi if isinstance(a, M.Midi.NoteEvent)]

    span('start', 'waveform_render')
    # Render the notes at the times specified
    waveform = render(notes, times, tone_len)
    span('end', 'waveform_render')

    # Save redered output
    filename = f'gen_melody{int(start_time)}.wav'
    librosa.output.write_wav(filename, waveform.numpy(), sr=sr, norm=True)
    logger(f"Saved waveform to: {filename}")

    # Record end time for rendering
    span('end', 'render_system')
