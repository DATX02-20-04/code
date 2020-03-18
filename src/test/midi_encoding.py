import tensorflow as tf
import preprocess as pre

dataset = tf.data.Dataset.from_tensor_slices(['test.midi'])

dataset = pre.pipeline([
    pre.midi(),
    pre.frame(400, 1),
    pre.unbatch(),
    pre.numpy(),
    pre.decode_midi(),
])(dataset)

x = next(dataset)

x.save('decoded.midi')
