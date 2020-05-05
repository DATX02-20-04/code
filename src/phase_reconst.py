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

# def cqt_spec(sr=16000, hop_length=512, n_bins=256, bins_per_octave=80, filter_scale=0.8, fmin=librosa.note_to_hz("C2")):<Paste>
#instrument = 4 # 4 = keyboard
#dataset = pro.filter(lambda x: x['instrument_family'] == instrument)(dataset)


def temp(x):
    mag = np.abs(x)
    phase = np.angle(x)
    mag = librosa.amplitude_to_db(mag)
    phase = pro.phase_to_inst_freq(phase)
    magphase = np.transpose([mag, phase], [1,2,0])
    return magphase

specs = pro.index_map('audio', pro.pipeline([
    #pro.cqt_spec(
    #   sr=hparams['sample_rate'], 
    #   bins_per_octave=40, 
    #   n_bins=256,
    #   filter_scale=5,
    #   fmin=librosa.note_to_hz("C1"),
    #   hop_length=256,
    #),
    pro.stft_spec(
        n_fft=1028,
        hop_length=256,
        win_length=1028
        ),
    #pro.pad([[0, 0], [0, 2], [0, 0]], 'CONSTANT', constant_values=hparams['log_amin']) 

]))(dataset)


for x in specs.take(1):
    audio = tf.unstack(x['audio'], axis=-1)
    mag = audio[0]
    phase = audio[1]
    print(f'img: {np.shape(mag)}')

    plt.subplot(1, 2, 1)
    plt.imshow(mag)

    plt.subplot(1, 2, 2)
    plt.imshow(phase)

    plt.savefig('stft_specs.png', bbox_inches='tight')

inverse = pro.index_map('audio', pro.pipeline([
    #pro.inverse_cqt_spec(
        #sr=hparams['sample_rate'], 
        #bins_per_octave=40, 
        #n_bins=256,
        #filter_scale=5,
        #fmin=librosa.note_to_hz("C1"),
        ##hop_length=256,
    pro.istft_spec(
        hop_length=256,
        win_length=1028
        )
]))(specs)


for x in inverse.take(1):
    librosa.output.write_wav('stft_specs.wav', x['audio'].numpy(), sr=hparams['sample_rate'])


