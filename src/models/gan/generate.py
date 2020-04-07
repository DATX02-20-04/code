import tensorflow as tf
from models.common.training import Trainer
from models.gan.model import GAN
import librosa
import matplotlib.pyplot as plt
import data.process as pro
import numpy as np


def start(hparams):
    gan_stats = np.load('gan_stats.npz')

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

    count = 10
    seed = tf.random.normal((count, hparams['latent_size']))
    mid = hparams['cond_vector_size']//2
    pitches = range(mid-count//2, mid+count//2)

    samples = tf.reshape(gan.generator([seed, pitches], training=False), [-1, 128, 128])
    x = tf.unstack(samples)
    plt.imshow(tf.concat(x, axis=1))
    plt.savefig('output.png')
    print(x)
    audio = pro.pipeline([
        pro.denormalize(normalization='specgan', stats=gan_stats),
        pro.invert_log_melspec(hparams['sample_rate'])
    ])(x)

    output = np.concatenate(list(audio))

    librosa.output.write_wav('gan_sample.wav', output, hparams['sample_rate'], norm=True)
