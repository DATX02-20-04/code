import tensorflow as tf
import librosa
import mido

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
        map_fn = fn if index_map is None else index_map(fn)
        if isinstance(dataset, tf.data.Dataset):
            return dataset.map(map_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            return map(map_fn, dataset)
    return transform

def composition_transform(transforms):
    def transform(dataset, index_map):
        for trns in transforms:
            dataset = trns(dataset, index_map)
        return dataset
    return transform

def filter_transform(fn):
    def transform(dataset, index_map):
        filter_fn = fn if index_map is None else index_map(fn)
        if isinstance(dataset, tf.data.Dataset):
            return dataset.filter(filter_fn)
        else:
            return filter(filter_fn, dataset)
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
    return composition_transform([
        stft(frame_length, frame_step, fft_length),
        abs(),
        transpose2d()
    ])

def melspec(sr, fft_length=1024, frame_step=512, frame_length=None, **kwargs):
    if frame_length is None:
        frame_length = fft_length
    return composition_transform([
        stft(frame_length, frame_step, fft_length),
        abs(),
        mels(sr, fft_length//2+1, **kwargs),
        transpose2d()
    ])

def invert_melspec(melspec, sr, fft_length=1024, frame_step=512, frame_length=None):
    if frame_length is None:
        frame_length = fft_length
    return librosa.feature.inverse.mel_to_audio(melspec, sr=sr, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)

def invert_log_melspec(melspec, sr, fft_length=1024, frame_step=512, frame_length=None, amin=1e-5):
    return composition_transform([
        log_to_amp(amin),
        invert_log_melspec(melspec, sr, fft_length, frame_step, frame_length)
    ])

def load_midi():
    return map_transform(lambda x: mido.MidiFile(x))

def encode_midi(note_count=128, max_time_shift=100, time_shift_m=10):
    return map_transform(_encode_midi(note_count, max_time_shift, time_shift_m))

def _encode_midi(note_count, max_time_shift, time_shift_m):
    def _midi(x):
        midi = []
        for msg in x:
            if not msg.is_meta:
                time_shift = min(int(msg.time*1000) // time_shift_ms, max_time_shift)-1
                time_enc = tf.reshape(tf.one_hot(np.array([time_shift]), max_time_shift), [-1])
                note = None
                etype = None

            if msg.type == 'note_on':
                note = tf.reshape(tf.one_hot(np.array([msg.note]), note_count), [-1])
                etype = [1]
            elif msg.type == 'note_off':
                note = tf.zeros((note_count,))
                etype = [0]

            if note is not None:
                midi.append(tf.concat([etype, note, time_enc], axis=0))
        return tf.stack(midi)
    return _midi

def midi(note_count=128, max_time_shift=100, time_shift_m=10):
    return composition_transform([
        load_midi(),
        encode_midi(note_count, max_time_shift, time_shift_m)
    ])


