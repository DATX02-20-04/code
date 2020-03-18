import os
import tensorflow as tf
import data.process as pro

def maestro_from_files(root_path, frame_size):
    dataset = tf.data.Dataset.list_files(os.path.join(root_path, '**/*.wav'))
    dataset = pro.pipeline(dataset, [
        pro.read_file(),
        pro.decode_wav(desired_channels=1),
        pro.map_transform(lambda x: x[0]),
        pro.reshape([-1]),
        pro.frame(frame_size, frame_size),
        pro.unbatch()
    ])

    return dataset
