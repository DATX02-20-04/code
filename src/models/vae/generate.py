import tensorflow as tf
from models.common.training import Trainer
from models.vae.model import VAE
import librosa


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

    samples = tf.reshape(vae.sample(100), [-1]).numpy()

    librosa.output.write_wav('vae_sample2.wav', samples, hparams['sample_rate'], norm=False)
