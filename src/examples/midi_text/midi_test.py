import tensorflow as tf
from .. import preprocess as pre


dataset = tf.data.Dataset.from_tensor_slices(["test.midi"])

dataset = pre.pipeline(
    [
        pre.midi(),
        pre.unbatch(),
        pre.prefetch(),
        pre.batch(100),
        pre.numpy(),
        pre.decode_midi(),
    ]
)(dataset)

x = next(dataset)

x.save("decoded.midi")
