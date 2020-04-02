import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
from models.common.training import Trainer
import data.process as pro
from models.transformer.model import Transformer

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(feature0):
    feature = {
        'midi': _int64_feature(feature0),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(f0):
    tf_string = tf.py_function(serialize_example, [f0], tf.string)
    return tf.reshape(tf_string, ())

def start(hparams):
    dataset = tf.data.Dataset.list_files('dataset/*.midi')

    dataset = pro.pipeline([
        pro.midi(),
        pro.prefetch(),
        pro.frame(hparams['frame_size'], hparams['frame_size'], True),
        pro.unbatch(),
        # pro.map_transform(tf_serialize_example)
    ])(dataset)

    def generator():
        for features in dataset:
            yield serialize_example(features)

    serialized_features_dataset = tf.data.Dataset.from_generator(generator,
                                                                 output_types=tf.string,
                                                                 output_shapes=())
    filename = 'midi_dataset.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)
