import wave
import tensorflow as tf
import numpy as np
import librosa

def generate(pitch, amp, vel, samples_per_note, sample_rate, seed=None):
    n = tf.shape(pitch)[0]
    t = tf.reshape(tf.tile(tf.linspace(0.0, samples_per_note/sample_rate, samples_per_note), [n]), [n, samples_per_note])
    print(t.shape, pitch.shape)
    return amp*tf.math.sin(t*2*np.pi*440*2**((-69+pitch)/12))*tf.math.exp(-t*(10/vel))

sr = 16000
samples_per_note = 8000
pitch_start = 60.0
n = 8
pitch = tf.reshape(tf.range(tf.cast(n, tf.float32)) + pitch_start, [n, 1])
vel = tf.ones([n, 1])*2
amp = tf.ones([n, 1])*0.5
note = generate(pitch, amp, vel, samples_per_note, sr)

librosa.output.write_wav('gen_tone.wav', note.numpy().flatten(), sr=sr)
