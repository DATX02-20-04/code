import tensorflow as tf
import preprocess

def nsynth_from_tfrecord(nsynth_tfrecord_path):
    dataset = tf.data.TFRecordDataset([nsynth_tfrecord_path])
    return preprocess.pipeline([
        preprocess.parse_tfrecord({
            "note_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64),
            "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64),
        }),
    ])(dataset)

def nsynth_to_melspec(dataset, hparams):
    if 'instrument' in hparams:
        dataset = preprocess.filter_transform(instrument_filter)(dataset)

    # Create preprocessing pipeline for the melspectograms
    return preprocess.pipeline([
        preprocess.extract('audio'),
        preprocess.melspec(sr=hparams['sample_rate']),
        preprocess.pad([[0, 0], [0, 2]], 'CONSTANT', constant_values=hparams['log_amin']),
        preprocess.amp_to_log(amin=hparams['log_amin']),
        preprocess.normalize(),
    ])(dataset)

instruments = {
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
