import tensorflow as tf
import numpy as np
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
from data.nsynth import instrument_families_filter
from models.sinenet.dataset import create_dataset
import librosa

def start(hparams):
    sinenet = SineNet(hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        model=sinenet,
    )

    trainer.init_checkpoint(ckpt)


    waves = []
    for p in range(0, 2):
        t = tf.linspace(0.0, 1.0, hparams['history'])
        hist = tf.math.sin(t*2*np.pi*440*2**(tf.cast(p, dtype=tf.float32)/12))
        hist = tf.reshape(hist, [1, hparams['history']])
        wave = hist

        for i in range(hparams['samples']):
            sample = sinenet.models[p](hist, training=False)
            _, hist = tf.split(hist, num_or_size_splits=[1, hist.shape[1]-1], axis=1)
            hist = tf.concat([hist, sample], axis=1)
            wave = tf.concat([wave, sample], axis=1)
            print(f"{i}/{hparams['samples']} pitch: {p}", end='\r')

    wave = tf.reshape(tf.stack(waves), [-1])

    librosa.output.write_wav('output.wav', wave.numpy(), sr=hparams['sample_rate'])
