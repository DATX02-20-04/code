import tensorflow as tf

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size, chan):
    return 1 - tf.linalg.band_part(tf.ones((size, chan)), -1, 0)
