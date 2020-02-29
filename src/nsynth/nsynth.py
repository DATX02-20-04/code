import tensorflow as tf
import numpy as np
import sys
import librosa
import matplotlib.pyplot as plt


class NSynthDataset():

    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path
        self.features = {
            "note_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64),
            "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64),
        }
        self.sr = 16000

    def dataset(self):
        self.examples = tf.data.TFRecordDataset([self.tfrecord_path])
        self.examples = self.examples.map(self.parse_example,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def one_hot(self, index, depth):
        self.examples = self.examples.map(self._map_index(index, lambda x: tf.one_hot(x, depth)),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def filter(self, predicate):
        self.examples = self.examples.filter(predicate)
        return self

    def extract(self, keys):
        self.examples = self.examples.map(lambda x: [x[key] for key in keys],
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def audio_padstart(self, index, timesteps):
        self.examples = self.examples.map(self._padstart(index, timesteps),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def _padstart(self, index, timesteps):
        def pad(x):
            padding = tf.random.normal([timesteps, 1])
            x[index] = tf.concat([padding, x[index]], 0)
            return x
        return pad

    def audio_windows(self, index, timesteps):
        self.examples = self.examples.map(
            self._map_index(index, lambda x: tf.signal.frame(x, timesteps, 1, axis=0)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.examples = self.examples.unbatch()
        return self

    def _map_index(self, index, fn):
        def imap(*x):
            return [fn(x[i]) if i == index else x[i] for i in range(len(x))]

        return imap

    def split(self, index, num_or_size_splits):
        self.examples = self.examples.map(
            self._map_index(index, lambda x: tf.split(x, num_or_size_splits, 0)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def stft(self, index, frame_length, hop_length, n_fft):
        # self.examples = self.examples.map(
        #     self._map_index(index, lambda x: tf.reshape(x, [1, -1])),
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.examples = self.examples.map(
            self._map_index(index, lambda x: tf.signal.stft(x,
                                                            frame_length=frame_length,
                                                            frame_step=hop_length,
                                                            fft_length=n_fft)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def abs(self, index):
        self.examples = self.examples.map(
            self._map_index(index, lambda x: tf.abs(x)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def melspec(self, index, n_mels):
        self.examples = self.examples.map(
            self._map_index(index, self._melspec(n_mels)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def set_chan(self, index):
        self.examples = self.examples.map(
            self._map_index(index, lambda x: tf.reshape(x, [*x.shape, 1])),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def _melspec(self, n_mels):
        def mel(x):
            num_spectrogram_bins = x.shape[-1]

            lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000, n_mels

            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, self.sr, lower_edge_hertz,
                upper_edge_hertz)

            return tf.tensordot(x, linear_to_mel_weight_matrix, 1)
        return mel

    def logspec(self, index):
        self.examples = self.examples.map(
            self._map_index(index, self._logspec),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def _logspec(self, x):
        amin = 1e-10
        logspec = tf.math.log(tf.math.maximum(amin, x))
        # logspec -= tf.math.log(tf.math.maximum(amin, ref))
        return logspec

    def flip_axes(self, index):
        self.examples = self.examples.map(
            self._map_index(index, lambda x: tf.transpose(x, [1, 0])),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def shuffle(self, buffer_size):
        self.examples = self.examples.shuffle(buffer_size)
        return self

    def batch(self, batch_size):
        self.examples = self.examples.batch(batch_size, drop_remainder=True)
        return self

    def unbatch(self):
        self.examples = self.examples.unbatch()
        return self

    def prefetch(self):
        self.examples = self.examples.prefetch(tf.data.experimental.AUTOTUNE)
        return self

    def parse_example(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.features)

