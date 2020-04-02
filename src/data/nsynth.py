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

def instrument_filter(instrument):
    def _filter(x):
        return tf.reshape(tf.math.equal(x['instrument_family'], instruments[instrument]), [])
    return _filter

def nsynth_to_melspec(dataset, hparams):
    if 'instrument' in hparams and hparams['instrument'] is not None:
        dataset = pro.filter_transform(instrument_filter(hparams['instrument']))(dataset)

    # Create preprocessing pipeline for the melspectograms
    return pro.pipeline([
        pro.extract('audio'),
        pro.melspec(sr=hparams['sample_rate']),
        pro.pad([[0, 0], [0, 4]], 'CONSTANT', constant_values=hparams['log_amin']),
        pro.amp_to_log(amin=hparams['log_amin']),
        pro.normalize(),
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
