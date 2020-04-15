# This script should be used to generate a waveform by first running the music
# transformer and then generating each individual note with specgan.
# This is then combined into a single wavfile.

# load MIDI dataset

# maybe the training script should save the checkpoints to drive

import util
import tensorflow as tf
import numpy as np
import librosa
import scripts.gen_tone
import models.transformer.generate as transformer
import models.gan.generate as gan
import data.process as pre
import data.midi as M

transformer_hparams = util.load_hparams('hparams/transformer.yml')
gan_hparams = util.load_hparams('hparams/gan.yml')

melody = transformer.generate(transformer_hparams)
midi = pre.decode_midi()(melody)
# with open("test.midi", "rb") as f:
# 	midi = M.read_midi(f)
midi = midi.flatten()[:80]

pitches = tf.cast([a.pitch for a in midi if isinstance(a, M.Midi.NoteEvent)], tf.int32)
amp     = tf.cast([a.velocity / 127 for a in midi if isinstance(a, M.Midi.NoteEvent)], tf.float32)
vel     = tf.ones_like(pitches)*2

sr = 16000
samples_per_note = 8000

def generate_tones(pitches):
	seed = tf.random.normal((len(pitches), gan_hparams['latent_size']))
	return gan.generate(gan_hparams, seed, pitches)[1]

notes = generate_tones(pitches)
notes = notes * amp[:,None]

times  = [int(a.time * samples_per_note) for a in midi if isinstance(a, M.Midi.NoteEvent)]
out = tf.zeros(max(times) + notes.shape[1])
for time, sound in zip(times, notes):
	out += tf.pad(sound, [(time, len(out)-len(sound)-time)])

librosa.output.write_wav('gen_melody.wav', out.numpy(), sr=sr, norm=True)
