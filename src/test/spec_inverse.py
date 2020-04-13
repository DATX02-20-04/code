import tensorflow as tf
import data.process as pro
from data.nsynth import nsynth_to_melspec
import matplotlib.pyplot as plt
import librosa
import numpy as np


dataset = tf.data.Dataset.list_files('src/audio/*.wav')
stats = np.load('gan_stats.npz')


hparams = {
    'sample_rate': 16000,
    'log_amin': 1e-6,
}

dataset = pro.pipeline([
    pro.wav(),
    pro.melspec(hparams['sample_rate'], n_mels=256, win_length=1024),
    pro.pad([[0, 0], [0, 2]], 'CONSTANT', constant_values=hparams['log_amin']),
    pro.normalize(normalization='specgan', stats=stats),
    pro.numpy(),
])(dataset)
x = next(dataset)
plt.imshow(tf.reverse(x, axis=[0]))
plt.axis('off')
plt.savefig('result.png')

dataset = pro.pipeline([
    pro.denormalize(normalization='specgan', stats=stats),
    pro.invert_log_melspec(hparams['sample_rate'], hop_length=256, win_length=1024)
])(dataset)

x = next(dataset)
librosa.output.write_wav('result.wav', x, hparams['sample_rate'])
