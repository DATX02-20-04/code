import tensorflow as tf
from models.common.training import Trainer
from models.testgan.model import TESTGAN
import librosa
import matplotlib.pyplot as plt
import data.process as pro
import numpy as np


def start(hparams):
    testgan_stats = np.load('testgan_stats.npz')

    testgan = TESTGAN((128, 128), hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=testgan.generator,
        selector=testgan.selector,
        gen_optimizer=testgan.generator_optimizer,
        disc_optimizer=testgan.selector_optimizer,
    )

    trainer.init_checkpoint(ckpt)

    count = 10
    seed = tf.random.normal((count, hparams['latent_size']))
    mid = hparams['cond_vector_size']//2
    pitches = tf.one_hot(range(mid-count//2, mid+count//2), hparams['cond_vector_size'])

    samples = tf.reshape(testgan.generator([seed, pitches], training=False), [-1, 128, 128])
    x = tf.unstack(samples)
    plt.imshow(tf.concat(x, axis=1))
    plt.show()
    print(x)
    audio = pro.pipeline([
        pro.denormalize(normalization='spectestgan', stats=testgan_stats),
        pro.invert_log_melspec(hparams['sample_rate'])
    ])(x)

    output = np.concatenate(list(audio))

    librosa.output.write_wav('testgan_sample.wav', output, hparams['sample_rate'], norm=True)
