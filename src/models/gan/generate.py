import tensorflow as tf
from models.common.training import Trainer
from models.gan.model import GAN
import librosa
import matplotlib.pyplot as plt
import data.process as pro
import numpy as np


def start(hparams):
    gan = GAN((128, 128), hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)

    seed = tf.random.normal((4, hparams['latent_size']))
    samples = tf.reshape(gan.generator(seed, training=False), [-1, 128, 128])
    plt.imshow(samples[0])
    plt.show()
    x = tf.unstack(samples)
    print(x)
    audio = pro.invert_log_melspec(hparams['sample_rate'])(x)

    output = np.concatenate(list(audio))

    librosa.output.write_wav('gan_sample.wav', output, hparams['sample_rate'], norm=False)
