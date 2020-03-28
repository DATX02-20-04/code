import tensorflow as tf
from models.common.training import Trainer
from models.vae.model import VAE
from scipy.io import wavfile


def start(hparams):
    vae = VAE(hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        encoder=vae.encoder,
        decoder=vae.decoder,
        vae=vae.vae,
    )

    trainer.init_checkpoint(ckpt)

    samples = tf.reshape(vae.sample(10), [-1]).numpy()

    wavfile.write('vae_sample2.wav', hparams['sample_rate'], samples)
