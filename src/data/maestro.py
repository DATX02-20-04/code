import os
import tensorflow as tf
import data.process as pro

def maestro_from_files(root_path, frame_size):
    dataset = tf.data.Dataset.list_files(os.path.join(root_path, '**/*.wav'))
    dataset = pro.pipeline([
        pro.read_file(),
        pro.decode_wav(desired_channels=1),
        pro.map_transform(lambda x: x[0]),
        pro.reshape([-1]),
        pro.frame(frame_size, frame_size),
        pro.unbatch()
    ])(dataset)

    return dataset


def maestro_to_melspec(dataset, hparams):
    """Turn maestro dataset into a dataset of melspectrograms."""
    return pro.pipeline([
        pro.melspec(sr=hparams['sample_rate']),
        pro.pad([[0, 0], [0, 4]], 'CONSTANT', constant_values=hparams['log_amin']),
        pro.amp_to_log(amin=hparams['log_amin']),
        pro.normalize(),
    ])(dataset)
