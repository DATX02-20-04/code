import tensorflow as tf
from models.common.training import Trainer
from models.vaegan.model import VAEGAN
import librosa


def start(hparams):
    vaegan = VAEGAN(hparams)

    trainer = Trainer(None, hparams)

    
    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        encoder=vaegan.encoder,
        decoder=vaegan.decoder,
        vaegan=vaegan.vaegan,
        critic=vaegan.critic
    )


    trainer.init_checkpoint(ckpt)

    samples = tf.reshape(vaegan.sample(100), [-1]).numpy()

    librosa.output.write_wav('vaegan_sample2.wav', samples, hparams['sample_rate'], norm=False)
