import tensorflow as tf
import preprocess as pre

dataset = tf.data.Dataset.list_files('/home/big/datasets/maestro-v2.0.0/**/*.midi')

dataset = pre.pipeline([
    pre.midi(),
    pre.unbatch(),
    pre.prefetch(),
])(dataset)

for x in dataset.take(10):
    print(x)
