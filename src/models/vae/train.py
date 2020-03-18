import tensorflow as tf
import tensorflow_probability as tfp
import data.process as preprocess
import os
from models.common.training import Trainer
from data.maestro import maestro_from_files
from models.vae.model import VAE
import matplotlib.pyplot as plt
#import IPython.display as display
import scipy.io.wavfile as wavfile
import tensorflow_datasets as tfds

tfpl = tfp.layers
tfd = tfp.distributions

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def start(hparams):
    # Load nsynth dataset
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)
    dataset = preprocess.pipeline([
        preprocess.extract('audio'),
        preprocess.frame(hparams['window_samples'], hparams['window_samples']),
        preprocess.unbatch(),
        preprocess.set_channels(1),
        preprocess.dupe(),
        preprocess.shuffle(hparams['buffer_size']),
        preprocess.batch(hparams['batch_size']),
        preprocess.prefetch()
    ])(dataset)

    vae = VAE(hparams)

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        encoder=vae.encoder,
        decoder=vae.decoder,
        vae=vae.vae,
    )

    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(vae.train_step)
    trainer.run()
