import tensorflow as tf
from models.common.training import Trainer
from models.gan.model import GAN
import librosa
import matplotlib.pyplot as plt
import data.process as pro
import numpy as np

def generate(hparams, seed, pitches):
    gan_stats = np.load('gan_stats.npz')

    gan = GAN((256, 128), hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)

    #seed = tf.repeat(seed, count, axis=0)
    pitches = tf.one_hot(pitches, hparams['cond_vector_size'], axis=1)

    samples = tf.reshape(gan.generator([seed, pitches], training=False), [-1, 256, 128])
    audio = pro.pipeline([
        pro.denormalize(normalization='specgan', stats=gan_stats),
        pro.invert_log_melspec(hparams['sample_rate']),
        list,     # Stupid workaround becuase invert_log_melspec only does
        np.array, # one spectrogram at a time
    ])(samples)
    return samples, audio

def start(hparams):
    count = 16
    seed = tf.random.normal((count, hparams['latent_size']))
    mid = hparams['cond_vector_size']//2
    pitches = range(mid-count//2, mid+count//2)
    samples, audio = generate(hparams, seed, pitches)

    width = 4
    height = 4
    plt.figure(figsize=(width*2, height*2))

    x = tf.unstack(samples)
    for i, img in enumerate(x):
        plt.subplot(width, height, i+1)
        plt.title(i)
        plt.imshow(tf.reverse(x[i], axis=[0]))
        plt.axis('off')

    plt.savefig('output.png')

    output = np.concatenate(list(audio))

    librosa.output.write_wav('gan_sample.wav', output, hparams['sample_rate'], norm=False)
