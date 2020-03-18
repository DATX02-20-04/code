import tensorflow as tf
import numpy as np

def get_angles(pos, i, d_model):
    w = 1.0 / np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos * w

def positional_encoding(position, d_model):
    a = get_angles(np.arange(position)[:, np.newaxis],
                   np.arange(d_model)[np.newaxis, :],
                   d_model)

    a[:, 0::2] = np.sin(a[:, 0::2])

    a[:, 1::2] = np.cos(a[:, 1::2])

    pos_enc = a[np.newaxis, ...]

    return tf.cast(pos_enc, dtype=tf.float32)
