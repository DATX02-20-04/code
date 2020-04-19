
import tensorflow as tf
import data.process as pro
from data.nsynth import nsynth_to_cqt_inst
import matplotlib.pyplot as plt
import matplotlib
import librosa
import numpy as np


dataset = tf.data.Dataset.list_files('src/audio/s_*.wav')
dataset = dataset.as_numpy_iterator()
dataset = list(dataset)
dataset = list(map(lambda x: x.decode('utf-8'), dataset))
dataset = sorted(dataset)


# stats = np.load('gan_stats.npz')


hparams = {
    'sample_rate': 16000,
    'log_amin': 0.000001,
    'cond_vector_size': 61
}

dataset = pro.pipeline([
    pro.wav()
])(dataset)

specs = nsynth_to_cqt_inst(dataset, hparams)

inverse = pro.pipeline([
    pro.inverse_cqt_spec()
])(specs)

S = next(inverse).numpy()
print(S, type(S))
librosa.output.write_wav('q_inverse.wav', S, sr=hparams['sample_rate'])
print("Done.")
# spec = next(inverse)
# plt.subplot(2, 2, 1)
# plt.imshow(spec[0])
# plt.subplot(2, 2, 2)
# plt.imshow(spec[1])
# plt.show()
