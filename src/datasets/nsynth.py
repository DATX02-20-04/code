import tensorflow as tf
import preprocess

def nsynth_from_tfrecord(nsynth_tfrecord_path):
    dataset = tf.data.TFRecordDataset([nsynth_tfrecord_path])
    return preprocess.pipeline(dataset, [
        preprocess.parse_tfrecord({
            "note_str": tf.io.FixedLenFeature([], dtype=tf.string),
            "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32),
            "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64),
            "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64),
        }),
    ])
