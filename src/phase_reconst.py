import tensorflow as tf
import tensorflow_datasets as tfds
import data.process as pro
from data.nsynth import nsynth_to_cqt_inst
import matplotlib.pyplot as plt
import matplotlib
import librosa
import numpy as np
from models.gan.train import calculate_dataset_stats

hparams = {
    'sample_rate': 16000,
    'log_amin': 0.000001,
    'cond_vector_size': 61
}

# Load nsynth dataset from tfds
dataset = tfds.load('nsynth/gansynth_subset', split='test', shuffle_files=True)

#gan_stats = calculate_dataset_stats(hparams, dataset)

specs = nsynth_to_cqt_inst(dataset, hparams)

for x in specs.take(1):
    audio = x['audio']
    mag = audio[:,:,0]
    phase = audio[:,:,1]
    print(x['pitch'])

    plt.subplot(1, 2, 1)
    plt.imshow(mag)

    plt.subplot(1, 2, 2)
    plt.imshow(phase)

    plt.savefig('recon.png')


inverse = pro.pipeline([
    pro.index_map('audio', pro.inverse_cqt_spec()),
])(specs)


for x in inverse.take(1):
    S = x['audio']
    librosa.output.write_wav('recon.wav', S.numpy(), sr=hparams['sample_rate'])
