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

hparams = util.load_hparams('hparams/transformer.yml')
melody = transformer.generate(hparams)
melody = melody.numpy()
melody = list(filter(lambda x: x < 128, melody))
melody = tf.cast(melody, tf.float32)

print(melody)

sr = 16000
samples_per_note = 8000
n = len(melody)
vel = tf.ones([n])*2
amp = tf.ones([n])*0.5
note = scripts.gen_tone.generate(melody, amp, vel, samples_per_note, sr)

notes = tf.reshape(note, [-1])
librosa.output.write_wav('gen_melody.wav', notes.numpy(), sr=sr)
