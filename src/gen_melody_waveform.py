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
import data.process as pre
import data.midi as M

hparams = util.load_hparams('hparams/transformer.yml')
melody = transformer.generate(hparams)
midi = pre.decode_midi()(melody)
# with open("test.midi", "rb") as f:
# 	midi = M.read_midi(f)
midi = midi.flatten()

pitches = [a.pitch for a in midi if isinstance(a, M.Midi.NoteEvent)]
amp     = [a.velocity / 127 for a in midi if isinstance(a, M.Midi.NoteEvent)]
pitches = tf.cast(pitches, tf.float32)
vel    = tf.ones_like(pitches)*2

sr = 16000
samples_per_note = 8000

times  = [int(a.time * samples_per_note) for a in midi if isinstance(a, M.Midi.NoteEvent)]

note = scripts.gen_tone.generate(pitches, amp, vel, samples_per_note*4, sr)

out = tf.zeros(max(times) + note.shape[1])
for time, sound in zip(times, note):
	out += tf.pad(sound, [(time, len(out)-len(sound)-time)])

librosa.output.write_wav('gen_melody.wav', out.numpy(), sr=sr, norm=True)
