import tensorflow as tf
import librosa
import numpy as np
import tensorflow_datasets as tfds
import data.process as pro
import matplotlib.pyplot as plt
from data.nsynth import instrument_families_filter, instrument_sources_filter

def serialize_example(x, y):
    feature = {
        'feature0': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(x, tf.float32)).numpy()])),
        'feature1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(y, tf.float32)).numpy()])),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def process(hparams, dataset):
    spec_dataset = pro.pipeline([
        pro.extract('audio'),
        pro.map_transform(lambda x: tf.py_function(lambda y: librosa.core.stft(y=y.numpy(),
                                                                               win_length=hparams['frame_length'],
                                                                               hop_length=hparams['frame_step'],
                                                                               n_fft=hparams['n_fft']),
                                                   [x], tf.complex64)),
    ])(dataset)

    mag_dataset = pro.pipeline([
        pro.map_transform(lambda x: tf.py_function(lambda S: librosa.feature.melspectrogram(S=S.numpy(),
                                                                                            win_length=hparams['frame_length'],
                                                                                            hop_length=hparams['frame_step'],
                                                                                            n_fft=hparams['n_fft'],
                                                                                            n_mels=hparams['n_mels']),
                                                   [x], tf.complex64)),
        pro.map_transform(lambda x: tf.transpose(x, [1, 0])),
        # pro.map_transform(lambda x: x[:, :-1]),
        pro.pad([[0, 2], [0, 0]], 'CONSTANT', constant_values=0),
        pro.abs(),
        pro.amp_to_log(),
    ])(spec_dataset)

    phase_dataset = pro.pipeline([
        pro.map_transform(lambda x: tf.transpose(x, [1, 0])),
        pro.map_transform(lambda x: x[:, :-1]),
        pro.pad([[0, 2], [0, 0]], 'CONSTANT', constant_values=0),
        pro.map_transform(lambda x: tf.numpy_function(np.angle, [x], tf.float32)),
        pro.map_transform(lambda x: tf.numpy_function(np.unwrap, [x], tf.float64)),
        pro.map_transform(lambda x: tf.cast(x, tf.float32)),
        pro.map_transform(lambda x: x[:, 1:] - x[:, :-1]),
        pro.map_transform(lambda x: tf.concat([x[:, 0:1], x], axis=1)),
        pro.map_transform(lambda x: x / np.pi),
    ])(spec_dataset)

    return tf.data.Dataset.zip((mag_dataset, phase_dataset))


def calculate_stats(hparams, dataset, examples):
    mag_means = []
    mag_stds = []
    mag_maxs = []
    mag_mins = []
    step = 0
    for mag, _ in dataset:
        # print(mag.shape)
        # exit()
        mag = mag.numpy()
        mag_means.append(np.mean(mag, axis=1))
        mag_stds.append(np.std(mag, axis=1))
        mag_maxs.append(np.max(mag))
        mag_mins.append(np.min(mag))
        step += 1
        print(f"Progress {int((step/examples)*100)}% | {step}/{examples}", end='\r')

    stats = {
        'examples': examples,
        'mag_mean': np.mean(mag_means, axis=0),
        'mag_std': np.mean(mag_stds, axis=0),
        'mag_max': np.max(mag_maxs),
        'mag_min': np.min(mag_mins),
    }

    np.savez(f"{hparams['dataset']}_stats.npz", **stats)

    return stats

def normalize(hparams, dataset, stats):
    dataset = pro.index_map(0, pro.pipeline([
        pro.normalize(normalization='neg_one_to_one', stats=stats),
        # pro.mels(hparams['sample_rate'], n_fft=hparams['frame_length']//2+1, n_mels=hparams['n_mels']),
    ]))(dataset)

    # y_dataset = pro.pipeline([
    #     pro.map_transform(lambda mag, phase: (tf.stack([mag, phase], axis=-1))),
    # ])(dataset)

    # x_dataset = pro.pipeline([
    #     pro.map_transform(lambda mag, phase: mag),
    #     pro.map_transform(lambda magphase: tf.reshape(magphase, [1, 128, 1024, 1])),
    #     pro.map_transform(lambda magphase: tf.image.resize(magphase, [128//(2**hparams['n_blocks']),
    #                                                                   1024//(2**hparams['n_blocks'])])),
    #     pro.map_transform(lambda magphase: tf.squeeze(magphase)),
    # ])(dataset)

    # dataset = pro.index_map(1, pro.pipeline([
    #     pro.mels(hparams['sample_rate'], n_fft=hparams['frame_length']//2+1, n_mels=hparams['n_mels']),
    # ]))(dataset)
    return dataset

def invert(hparams, stats):
    return pro.pipeline([
        pro.index_map(0, pro.pipeline([
            pro.denormalize(normalization='neg_one_to_one', stats=stats),
            pro.log_to_amp(),
            pro.cast(tf.complex64),
        ])),
        pro.index_map(1, pro.pipeline([
            pro.map_transform(lambda x: x * np.pi),
            pro.map_transform(lambda x: tf.math.cumsum(x, axis=0)),
            pro.map_transform(lambda x: (x + np.pi) % (2 * np.pi) - np.pi),
            pro.cast(tf.complex64),
        ])),
        pro.pipeline([
            pro.map_transform(lambda mag, phase: mag * tf.math.exp(1.0j * phase)),
            pro.map_transform(lambda spec: tf.signal.inverse_stft(spec,
                                                                  frame_length=hparams['frame_length'],
                                                                  frame_step=hparams['frame_step'],
                                                                  window_fn=tf.signal.inverse_stft_window_fn(hparams['frame_step']))),
        ])
    ])

def start(hparams):
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=False)

    if 'instrument' in hparams and hparams['instrument'] is not None:
        instrument = hparams['instrument']
        if 'family' in instrument and instrument['family'] is not None:
            dataset = pro.filter(instrument_families_filter(instrument['family']))(dataset)
            print("FILTER FAMILY", instrument['family'])
        if 'source' in hparams and hparams['source'] is not None:
            dataset = pro.filter(instrument_sources_filter(instrument['source']))(dataset)
    if 'max_examples' in hparams:
        dataset = dataset.take(hparams['max_examples'])

    print("Counting examples...")
    examples = 0
    for features in dataset:
        examples += 1
        print(f"{examples}", end='\r')
    print(f"Counted {examples} examples.")

    # Process dataset examples
    dataset = process(hparams, dataset).cache()
    # dataset = dataset.cache(filename='preprocess_cache')

    print("Calculating dataset stats...")
    stats = calculate_stats(hparams, dataset, examples)
    print(stats['mag_mean'].shape)
    print("Calculating dataset stats done.")

    dataset = normalize(hparams, dataset, stats)

    # Create generator for serializing the examples
    def generator():
        step = 0
        for features in dataset:
            print(f"Progress {int((step/examples)*100)}% | {step}/{examples}", end='\r')
            step += 1
            yield serialize_example(*features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())

    print("Preprocessing...")

    filename = f"{hparams['dataset']}_dataset.tfrecord"
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

    print("\nDONE")


def load(hparams):
    stats = np.load(f"{hparams['dataset']}_stats.npz")
    dataset = tf.data.TFRecordDataset([f"{hparams['dataset']}_dataset.tfrecord"])
    return pro.pipeline([
        pro.parse_tfrecord({
            'feature0': tf.io.FixedLenFeature([], dtype=tf.string),
            'feature1': tf.io.FixedLenFeature([], dtype=tf.string),
        }),
        pro.map_transform(lambda x: (tf.io.parse_tensor(x['feature0'], out_type=tf.float32),
                                     tf.io.parse_tensor(x['feature1'], out_type=tf.float32)))
    ])(dataset), stats

