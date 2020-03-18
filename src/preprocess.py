import tensorflow as tf
import librosa
import mido
import math
import numpy as np
import itertools

def index_map(index, f):
    # Carl: I don't think this parallelizes very well, but I'm not sure
    def imap(x):
        x[index] = f(x[index])
        return x
    return map_transform(imap)

def pipeline(transforms):
    def transform(dataset):
        for trns in transforms:
            dataset = trns(dataset)
        return dataset
    return transform

def map_transform(fn):
    def transform(dataset):
        if isinstance(dataset, tf.data.Dataset):
            return dataset.map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            return map(fn, dataset)
    return transform

def numpy():
    def transform(dataset):
        if isinstance(dataset, tf.data.Dataset):
            return dataset.as_numpy_iterator()
        else:
            return dataset
    return transform

def tensor(output_types):
    def transform(dataset):
        if isinstance(dataset, tf.data.Dataset):
            return dataset
        else:
            def _generator():
                for x in dataset:
                    yield x
            return tf.data.Dataset.from_generator(_generator, output_types)
    return transform

def filter_transform(fn):
    def transform(dataset):
        if isinstance(dataset, tf.data.Dataset):
            return dataset.filter(fn)
        else:
            return filter(fn, dataset)
    return transform


#
# TRANSFORMS
#
def parse_tfrecord(features):
    return map_transform(lambda x: tf.io.parse_single_example(x, features))

def read_file():
    return map_transform(lambda x: tf.io.read_file(x))

def decode_wav(desired_channels=-1, desired_samples=-1):
    return map_transform(lambda x: tf.audio.decode_wav(x, desired_channels, desired_samples))

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
    return lambda dataset: dataset.batch(batch_size)

def unbatch():
    return lambda dataset: dataset.unbatch()

def shuffle(buffer_size):
    return lambda dataset: dataset.shuffle(buffer_size)

def prefetch():
    return lambda dataset: dataset.prefetch(tf.data.experimental.AUTOTUNE)

def pad(paddings, mode, constant_values=0, name=None):
    return map_transform(_pad(paddings, mode, constant_values, name))

def _pad(paddings, mode, constant_values=0, name=None):
    def _p(x):
        if type(constant_values) == str:
            if constant_values == 'min':
                values = tf.reduce_min(x)
            elif constant_values == 'max':
                values = tf.reduce_max(x)
        else:
            values = constant_values
        return tf.pad(x, paddings, mode, values, name)
    return _p


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

def dupe():
    return map_transform(lambda x: (x, x))

def _normalize(normalization='neg_one_to_one'):
    def _n(x):
        _max = tf.reduce_max(x)
        _min = tf.reduce_min(x)
        if normalization == 'neg_one_to_one':
            return ((x - _min) / (_max - _min)) * 2 - 1
        elif normalization == 'zero_to_one':
            return ((x - _min) / (_max - _min))
    return _n

def normalize(normalization='neg_one_to_one'):
    return map_transform(_normalize(normalization))

def amp_to_log(amin=1e-5):
    return map_transform(lambda x: tf.math.log(x + amin))

def log_to_amp(amin=1e-5):
    return map_transform(lambda x: tf.math.exp(x) - amin)

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

def spec(fft_length=1024, frame_step=512, frame_length=None, **kwargs):
    if frame_length is None:
        frame_length = fft_length
    return pipeline([
        stft(frame_length, frame_step, fft_length),
        abs(),
        transpose2d()
    ])

def melspec(sr, n_fft=1024, hop_length=512, win_length=None, **kwargs):
    return pipeline([
        numpy(),
        map_transform(lambda x: librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)),
        map_transform(lambda x: librosa.core.power_to_db(x, ref=1.0)),
        tensor(tf.float32),
        # stft(frame_length, frame_step, fft_length),
        # abs(),
        # mels(sr, fft_length//2+1, **kwargs),
        # transpose2d()
    ])

def denormalize(denorm_amin=-20, denorm_amax=0, normalization='neg_one_to_one'):
    if normalization == 'neg_one_to_one':
        return map_transform(lambda x: (((x+1)*0.5)*(denorm_amax-denorm_amin)+denorm_amin))
    elif normalization == 'zero_to_one':
        return map_transform(lambda x: (x*(denorm_amax-denorm_amin)+denorm_amin))
    else:
        raise Exception(f"No normalization type named '{normalization}'.")

def invert_log_melspec(sr, n_fft=1024, hop_length=512, win_length=None, amin=1e-5, denorm_amin=-38, denorm_amax=0):
    return pipeline([
        # denormalize(denorm_amin, denorm_amax),
        # log_to_amp(amin),
        map_transform(lambda x: librosa.core.db_to_power(x, ref=1.0)),
        map_transform(lambda x: librosa.feature.inverse.mel_to_audio(x, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    ])

def load_midi():
    return map_transform(lambda x: mido.MidiFile(x))

def encode_midi(note_count=128, max_time_shift=100, time_shift_ms=10, velocity_count=100):
    def _midi(x):
        midi = []
        for msg in x:
            if not msg.is_meta:
                time_shift = min(int(msg.time*1000) // time_shift_ms, max_time_shift)-1
                time_enc = tf.reshape(tf.one_hot(np.array([time_shift]), max_time_shift), [-1])
                note = None
                velocity = None
                etype = None

                if msg.type == 'note_on':
                    note = tf.reshape(tf.one_hot(np.array([msg.note]), note_count), [-1])
                    velocity = tf.reshape(tf.one_hot(np.array([msg.velocity]), velocity_count), [-1])
                    etype = [1]
                else:# msg.type == 'note_off':
                    note = tf.zeros((note_count,))
                    velocity = tf.zeros((velocity_count,))
                    etype = [0]

                if note is not None:
                    midi.append(tf.concat([etype, note, velocity, time_enc], axis=0))

        return tf.stack(midi)
    return map_transform(_midi)

def decode_midi(note_count=128, max_time_shift=100, time_shift_ms=10, velocity_count=100):
    def _midi(x):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        for e in tf.unstack(x):
            [mtype, note, velocity, time] = tf.split(e, [1, note_count, velocity_count, max_time_shift])
            if mtype == 1:
                mtype = 'note_on'
            else:
                mtype = 'note_off'

            note = tf.argmax(note, axis=-1).numpy()
            velocity = tf.argmax(velocity, axis=-1).numpy()
            time = tf.argmax(time, axis=-1)
            time = ((time+1)*time_shift_ms)/1000
            time = int(round(mido.second2tick(time.numpy(), mid.ticks_per_beat, 500000)))

            if mtype == 'note_on':
                track.append(mido.Message(mtype, note=note, velocity=velocity, time=time))
            else:
                track.append(mido.Message(mtype, time=time))

        return mid
    return map_transform(_midi)

def midi(note_count=128, max_time_shift=100, time_shift_m=10):
    return pipeline([
        numpy(),
        load_midi(),
        encode_midi(note_count, max_time_shift, time_shift_m),
        tensor(tf.float32),
    ])
