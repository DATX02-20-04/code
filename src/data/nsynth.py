import tensorflow as tf
import data.process as pro

def nsynth_from_tfrecord(nsynth_tfrecord_path):
    dataset = tf.data.TFRecordDataset([nsynth_tfrecord_path])
    return pro.pipeline([
        pro.parse_tfrecord({
            "note_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64),
            "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64),
        }),
    ])(dataset)

def instrument_filter(key, value, value_map):
    def _filter(x):
        return tf.reshape(tf.math.equal(x['instrument'][key], value_map[value]), [])
    return _filter

def instrument_sources_filter(value):
    return instrument_filter('source', value, instrument_sources)

def instrument_families_filter(value):
    return instrument_filter('family', value, instrument_families)

def nsynth_to_melspec(dataset, hparams, stats=None):
    if 'instrument' in hparams and hparams['instrument'] is not None:
        instrument = hparams['instrument']
        if 'family' in instrument and instrument['family'] is not None:
            dataset = pro.filter(instrument_families_filter(instrument['family']))(dataset)
        if 'source' in hparams and hparams['source'] is not None:
            dataset = pro.filter(instrument_sources_filter(instrument['source']))(dataset)

    dataset = pro.index_map('pitch', pro.one_hot(hparams['cond_vector_size']))(dataset)

    dataset = pro.index_map('audio', pro.pipeline([
        pro.melspec(sr=hparams['sample_rate']),
        pro.pad([[0, 0], [0, 2]], 'CONSTANT', constant_values=hparams['log_amin']),
    ]))(dataset)

    if stats is not None:
        dataset = pro.index_map('audio', pro.normalize(normalization='specgan', stats=stats))(dataset)

    # Create preprocessing pipeline for the melspectograms
    return dataset

instrument_families = {
    'bass': 0,
    'brass': 1,
    'flute': 2,
    'guitar': 3,
    'keyboard': 4,
    'mallet': 5,
    'organ': 6,
    'reed': 7,
    'string': 8,
    'synth_lead': 9,
    'vocal': 10
}

instrument_sources = {
    'acoustic': 0,
    'electronic': 1,
    'synthetic': 2,
}
