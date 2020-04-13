import tensorflow as tf
import numpy as np

def create_dataset(hparams, hist_len=None):
    if hist_len is None:
        hist_len = hparams['history']

    def gen_pitches():
        while True:
            yield tf.random.uniform([1], 0, 128, dtype=tf.int32)

    def sinewave(pitch):
        length = hist_len+1
        t = tf.linspace(0.0, 1.0, length)
        return tf.math.sin(t*2*np.pi*440*2**(tf.cast(-69+pitch, dtype=tf.float32)/12))

    pitch_dataset = tf.data.Dataset.from_generator(gen_pitches, output_types=tf.int32)

    wave_dataset = pitch_dataset.map(sinewave)
    pitch_enc_dataset = pitch_dataset.map(lambda x: tf.reshape(tf.one_hot(x, hparams['pitches']), [-1]))
    next_dataset = wave_dataset.map(lambda x: x[-1])
    hist_dataset = wave_dataset.map(lambda x: x[:-1])

    dataset = tf.data.Dataset.zip((pitch_enc_dataset, hist_dataset, next_dataset))

    return dataset
