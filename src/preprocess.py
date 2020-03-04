import tensorflow as tf
import librosa

def index_map(index):
    def map_index(fn):
        def imap(x):
            x[index] = fn(x[index])
            return x
        return imap
    return map_index

def pipeline(dataset, transforms, index_map=None):
    for transform in transforms:
        dataset = transform(dataset, index_map)
    return dataset

def map_transform(fn):
    def transform(dataset, index_map):
        return dataset.map(fn if index_map is None else index_map(fn),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return transform

def composition_transform(transforms):
    def transform(dataset, index_map):
        for trns in transforms:
            dataset = trns(dataset, index_map)
        return dataset
    return transform

def filter_transform(predicate):
    def transform(dataset, index_map):
        return dataset.filter(fn if index_map is None else index_map(fn),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return transform


#
# TRANSFORMS
#
def parse_tfrecord(features):
    return map_transform(lambda x: tf.io.parse_single_example(x, features))

def one_hot(depth):
    return map_transform(lambda x: tf.one_hot(x, depth))

def filter(predicate):
    return filter_transform(predicate)

def extract(key):
    return map_transform(lambda x: x[key])

def reshape(shape):
    return map_transform(lambda x: tf.reshape(x, shape))

def set_channels(channels):
    return map_transform(lambda x: tf.reshape(x, [*x.shape, channels]))

def batch(batch_size):
    return lambda dataset, index_map: dataset.batch(batch_size)

def unbatch():
    return lambda dataset, index_map: dataset.unbatch()

def shuffle(buffer_size):
    return lambda dataset, index_map: dataset.shuffle(buffer_size)

def prefetch():
    return lambda dataset, index_map: dataset.prefetch(tf.data.experimental.AUTOTUNE)

def pad(paddings, mode, constant_values=0, name=None):
    return map_transform(lambda x: tf.pad(x, paddings, mode, constant_values, name))

def frame(frame_length, frame_step, pad_end=False, pad_value=0, axis=-1, name=None):
    return map_transform(lambda x: tf.signal.frame(x, frame_length,
                                                          frame_step, pad_end,
                                                          pad_value, axis, name))

def split(num_or_size_splits, axis=0, num=None, name='split'):
    return map_transform(lambda x: tf.split(x, num_or_size_splits,
                                                   axis, num, name))

def stft(frame_length, frame_step, fft_length=None):
    return map_transform(lambda x: tf.signal.stft(x, frame_length,
                                                  frame_step, fft_length))

def istft(frame_length, frame_step, fft_length=None):
    return map_transform(lambda x: tf.signal.inverse_stft(x, frame_length,
                                                          frame_step, fft_length))

def abs():
    return map_transform(lambda x: tf.abs(x))

def _normalize(x):
    _max = tf.reduce_max(x)
    _min = tf.reduce_min(x)
    return ((x - _min) / (_max - _min)) * 2 - 1

def normalize():
    return map_transform(_normalize)

def amp_to_log(amin=1e-5):
    return map_transform(lambda x: tf.math.log(tf.maximum(amin, x)))

def log_to_amp():
    return map_transform(lambda x: tf.math.exp(x))

def log_to_amp():
    return map_transform(lambda x: tf.math.exp(x))

def mels(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    def mel(x):
        #fft_length = x.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            n_mels, n_fft, sr, fmin, fmax)

        return tf.tensordot(x, linear_to_mel_weight_matrix, 1)
    return map_transform(mel)

def transpose2d():
    return map_transform(lambda x: tf.transpose(x, [1, 0]))

def spec(fft_length=2048, frame_step=512, frame_length=None, **kwargs):
    if frame_length is None:
        frame_length = fft_length
    return composition_transform([
        stft(frame_length, frame_step, fft_length),
        abs(),
        transpose2d()
    ])

def melspec(sr, fft_length=2048, frame_step=512, frame_length=None, **kwargs):
    if frame_length is None:
        frame_length = fft_length
    return composition_transform([
        stft(frame_length, frame_step, fft_length),
        abs(),
        mels(sr, fft_length//2+1, **kwargs),
        transpose2d()
    ])
