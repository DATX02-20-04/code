import tensorflow as tf
import numpy as np

def create_dataset(hparams, hist_len=None):
    if hist_len is None:
        hist_len = hparams['history']

    def sinewave():
        while True:
            length = hist_len+1
            t = tf.linspace(0.0, 1.0, length)
            yield [tf.math.sin(t*2*np.pi*440*2**(tf.cast(pitch, dtype=tf.float32)/12)) for pitch in range(hparams['pitches'])]

    wave_dataset = tf.data.Dataset.from_generator(sinewave, output_types=tf.float32)

    next_dataset = wave_dataset.map(lambda xs: xs[:, -1])
    hist_dataset = wave_dataset.map(lambda xs: xs[:, :-1]+tf.random.normal(tf.shape(xs[:, :-1]))*0.01)

    dataset = tf.data.Dataset.zip((hist_dataset, next_dataset))

    return dataset
