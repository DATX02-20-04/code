import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class BatchStd(tfkl.Layer):
    def __init__(self):
        super(BatchStd, self).__init__()

    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=0, keepdims=True)

        squared_mean_dists = (x - mean)**2
        mean_dist = tf.math.reduce_mean(squared_mean_dists, axis=0, keepdims=True)
        mean_dist += 1e-8

        std = tf.math.sqrt(mean_dist)
        mean_std = tf.math.reduce_mean(std, keepdims=True)

        shape = tf.shape(x)

        y = tf.tile(mean_std, (shape[0], shape[1], 1))

        return tf.concat([x, y], axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class WeightedSum(tfkl.Add):
    def __init__(self, alpha=0.0):
        super(WeightedSum, self).__init__()
        self.alpha = tf.Variable(alpha, name='alpha')

    def _merge_function(self, x):
        assert (len(x) == 2)

        return ((1.0 - self.alpha) * x[0]) + (self.alpha * x[1])


class PixelNorm(tfkl.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def call(self, x):
        values = x**2.0
        mean_values = tf.math.reduce_mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8

        l2 = tf.math.sqrt(mean_values)

        return x / l2

    def compute_output_shape(self, input_shape):
        return input_shape
