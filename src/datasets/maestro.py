import os
import tensorflow as tf
import preprocess

def maestro_from_files(root_path, frame_size):
    dataset = tf.data.Dataset.list_files(os.path.join(root_path, '**/*.wav'))
    dataset = preprocess.pipeline(dataset, [
        preprocess.read_file(),
        preprocess.decode_wav(desired_channels=1),
        preprocess.map_transform(lambda x: x[0]),
        preprocess.reshape([-1]),
        preprocess.frame(frame_size, frame_size),
        preprocess.unbatch()
    ])

    return dataset
