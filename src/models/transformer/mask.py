import tensorflow as tf

''' Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input.
    The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.'''

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
