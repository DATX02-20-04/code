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

def resample(orig_sr, target_sr, dtype=None):
    return map_transform(lambda x: tf.reshape(tf.py_function(lambda x: librosa.core.resample(x.numpy(), orig_sr=orig_sr, target_sr=target_sr), [tf.reshape(x, [-1])], x.dtype if dtype is None else dtype), [-1]))

def read_file():
    return map_transform(lambda x: tf.io.read_file(x))

def decode_wav(desired_channels=-1, desired_samples=-1):
    return map_transform(lambda x: tf.audio.decode_wav(x, desired_channels, desired_samples))

def wav(desired_channels=-1, desired_samples=-1):
    return pipeline([
        read_file(),
        decode_wav(desired_channels, desired_samples),
        map_transform(lambda x: x[0]),
        reshape([-1]),
    ])

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

def cache(filename=''):
    return lambda dataset: dataset.cache(filename)

def batch(batch_size, drop_remainder=False):
    return lambda dataset: dataset.batch(batch_size, drop_remainder)

def unbatch():
    return lambda dataset: dataset.unbatch()

def shuffle(buffer_size):
    return lambda dataset: dataset.shuffle(buffer_size)

def prefetch():
    return lambda dataset: dataset.prefetch(tf.data.experimental.AUTOTUNE)

def pad(paddings, mode, constant_values=0, name=None):
    return map_transform(lambda x: tf.pad(x, paddings, mode, constant_values, name))

def frame(frame_length, frame_step, pad_end=False, pad_value=0, axis=-1, name=None):
    return map_transform(lambda x: tf.signal.frame(x, frame_length,
                                                          frame_step, pad_end,
                                                          pad_value, axis, name))

def split(num_or_size_splits, axis=0, num=None, name='split'):
    return map_transform(lambda x: tf.split(x, num_or_size_splits,
                                                   axis, num, name))

def debug():
    return map_transform(lambda x: tf.py_function(_debug, [x], x.dtype))

def _debug(x):
    print(x)
    return x

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

def _normalize(x):
    _max = tf.reduce_max(x)
    _min = tf.reduce_min(x)
    return ((x - _min) / (_max - _min)) * 2 - 1

def normalize():
    return map_transform(_normalize)

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

def melspec(sr, fft_length=1024, frame_step=512, frame_length=None, **kwargs):
    if frame_length is None:
        frame_length = fft_length
    return pipeline([
        stft(frame_length, frame_step, fft_length),
        abs(),
        mels(sr, fft_length//2+1, **kwargs),
        transpose2d()
    ])

def invert_melspec(sr, fft_length=1024, frame_step=512, frame_length=None):
    if frame_length is None:
        frame_length = fft_length
    return map_transform(lambda x: librosa.feature.inverse.mel_to_audio(x.numpy(), sr=sr, n_fft=fft_length, hop_length=frame_step, win_length=frame_length))

def invert_log_melspec(sr, fft_length=1024, frame_step=512, frame_length=None, amin=1e-5):
    return pipeline([
        log_to_amp(amin),
        invert_melspec(sr, fft_length, frame_step, frame_length)
    ])

def load_midi():
    return map_transform(lambda x: mido.MidiFile(x))


def encode_midi(note_count=128, time_shift_count=100, time_shift_ms=10, velocity_count=100):
    _event_start = {
        'note_on': 0,
        'note_off': note_count,
        'time_shift': note_count * 2,
        'velocity': note_count * 2 + time_shift_count,
    }

    def _midi(x):
        midi = []
        prev_note = 0
        for msg in x:
            if not msg.is_meta:
                time_shift = min(int(msg.time*1000) // time_shift_ms, time_shift_count-1)

                if msg.type == 'note_on':
                    midi.append(min(msg.note, note_count-1) + _event_start['note_on'])
                    midi.append(min(msg.velocity, velocity_count-1) + _event_start['velocity'])
                    prev_note = msg.note
                else:# msg.type == 'note_off':
                    midi.append(min(prev_note, note_count-1) + _event_start['note_off'])

                midi.append(time_shift+_event_start['time_shift'])

        return tf.cast(tf.stack(midi), dtype=tf.int64)
    return map_transform(_midi)

def decode_midi(note_count=128, time_shift_count=100, time_shift_ms=10, velocity_count=100):
    note_on_range = range(0, note_count)
    note_off_range = range(note_count, note_count*2)
    time_shift_range = range(note_count*2, note_count*2+time_shift_count)

    def _midi(x):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        msgs = []
        prev_msg = None
        for e in x.numpy():
            print(e)
            e = int(e)
            print(e)
            if e in note_on_range:
                msg = {'type': 'note_on', 'note': e}
                msgs.append(msg)
                prev_msg = msg
            elif e in note_off_range:
                msg = {'type': 'note_off'}
                msgs.append(msg)
                prev_msg = msg
            elif e in time_shift_range and prev_msg is not None:
                prev_msg['time_shift'] = e - note_count*2
            elif prev_msg is not None:
                prev_msg['velocity'] = e - note_count*2 - time_shift_count

        print(msgs)
        note_on = False
        for msg in msgs:
            if msg['type'] == 'note_on':
                # if 'velocity' not in msg and 'time_shift' not in msg:
                #     break
                track.append(mido.Message('note_on',
                                          note=msg['note'],
                                          velocity=msg['velocity'] if 'velocity' in msg else 20,
                                          time=msg['time_shift']*time_shift_ms if 'time_shift' in msg else 50))
                note_on = True
            else:
                # if 'time_shift' not in msg:
                #     break
                track.append(mido.Message('note_off',
                                          time=msg['time_shift']*time_shift_ms if 'time_shift' in msg else 50))
                note_on = False
        print(track)

        return mid
    return map_transform(_midi)

def midi(note_count=128, max_time_shift=100, time_shift_m=10):
    return pipeline([
        numpy(),
        load_midi(),
        encode_midi(note_count, max_time_shift, time_shift_m),
        tensor(tf.int64),
    ])
