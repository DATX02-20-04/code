import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
from data.nsynth import instrument_families_filter
from models.sinenet.dataset import create_dataset
import librosa

def start(hparams):
    dataset = create_dataset(hparams, 16000)
    dataset = pro.pipeline([
        pro.batch(16),
    ])(dataset).as_numpy_iterator()

    samples = next(dataset)[1].flatten()

    librosa.output.write_wav('dataset_samples.wav', samples, sr=hparams['sample_rate'])
    print("Saved dataset samples.")

    sinenet = SineNet(hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        model=sinenet.model,
    )

    trainer.init_checkpoint(ckpt)

    pitch = tf.one_hot(tf.range(40, 80), hparams['pitches'])
    hist = tf.random.normal([pitch.shape[0], hparams['history']])
    wave = tf.zeros([pitch.shape[0], 0])

    for i in range(hparams['samples']):
        sample = sinenet.model([pitch, hist], training=False)
        _, hist = tf.split(hist, num_or_size_splits=[1, hist.shape[1]-1], axis=1)
        hist = tf.concat([hist, sample], axis=1)
        wave = tf.concat([wave, sample], axis=1)
        print(f"{i}/{hparams['samples']}", end='\r')

    wave = tf.reshape(wave, [-1])

    librosa.output.write_wav('output.wav', wave.numpy(), sr=hparams['sample_rate'])
