import tensorflow as tf
import data.process as pro
from data.nsynth import nsynth_to_melspec
import matplotlib.pyplot as plt
import librosa


dataset = tf.data.Dataset.list_files('test.wav')


hparams = {
    'sample_rate': 16000,
    'log_amin': 1e-6,
}

dataset = pro.pipeline([
    pro.wav(),
    pro.melspec(hparams['sample_rate']),
    pro.numpy(),
])(dataset)

dataset = pro.pipeline([
    pro.invert_log_melspec(hparams['sample_rate'])
])(dataset)

x = next(dataset)
librosa.output.write_wav('result.wav', x, hparams['sample_rate'])
