import wave
import tensorflow as tf
import numpy as np
import librosa

def generate(pitch, amp, vel, samples_per_note, sample_rate, seed=None):
    n = tf.shape(pitch)[0]
    t = tf.reshape(tf.tile(tf.linspace(0.0, samples_per_note/sample_rate, samples_per_note), [n]), [n, samples_per_note])
    pitch = tf.reshape(pitch, [-1, 1])
    amp = tf.reshape(amp, [-1, 1])
    vel = tf.reshape(vel, [-1, 1])
    return amp*tf.math.sin(t*2*np.pi*440*2**((-69+pitch)/12))*tf.math.exp(-t*(10/vel))

