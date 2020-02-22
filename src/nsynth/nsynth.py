import tensorflow as tf
import numpy as np
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

    def filter(self, predicate):
        self.examples = self.examples.filter(predicate)
        return self

    def audio(self):
        self.examples = self.examples.map(lambda x: tf.reshape(x['audio'], [-1, 1]),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def audio_padstart(self, timesteps):
        self.examples = self.examples.map(self._padstart(timesteps),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def _padstart(self, timesteps):
        def pad(x):
            padding = tf.random.normal([timesteps, 1])
            return tf.concat([padding, x], 0)
        return pad

    def audio_windows(self, timesteps):
        self.examples = self.examples.map(lambda x: tf.signal.frame(x, timesteps, 1, axis=0),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.examples = self.examples.unbatch()
        return self

    def audio_split(self, num_or_size_splits):
        self.examples = self.examples.map(lambda x: tf.split(x, num_or_size_splits, 0),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def audio_stft(self):
        self.examples = self.examples.map(lambda x: tf.reshape(x, [1, -1]),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.examples = self.examples.map(lambda x: tf.signal.stft(x,
                                                                   frame_length=1024,
                                                                   frame_step=512,
                                                                   fft_length=1024),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def stft_magspec(self):
        self.examples = self.examples.map(lambda x: tf.abs(x),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def magspec_melspec(self):
        self.examples = self.examples.map(lambda x: self._melspec(x),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def _melspec(self, x):
        num_spectrogram_bins = x.shape[-1]

        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000, 128

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, self.sr, lower_edge_hertz,
            upper_edge_hertz)

        return tf.tensordot(x, linear_to_mel_weight_matrix, 1)

    def shuffle(self, buffer_size):
        self.examples = self.examples.shuffle(buffer_size)
        return self

    def batch(self, batch_size):
        self.examples = self.examples.batch(batch_size)
        return self

    def prefetch(self):
        self.examples = self.examples.prefetch(tf.data.experimental.AUTOTUNE)
        return self

    def parse_example(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.features)

