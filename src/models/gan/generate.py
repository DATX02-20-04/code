import tensorflow as tf
from models.common.training import Trainer
from models.gan.model import GAN
import librosa
import matplotlib.pyplot as plt
import data.process as pro
import numpy as np


def start(hparams):
    gan_stats = np.load('gan_stats.npz')

    gan = GAN((128, 256), hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)

    count = 8

    # Random seed for each
    seed = tf.random.normal((count, hparams['latent_size']))

    # Same seed for all
    # seed = tf.random.normal((1, hparams['latent_size']))
    # seed = tf.repeat(seed, count, axis=0)

    # Sample pitch from middle
    mid = hparams['cond_vector_size']//2
    pitches = range(mid-count//2, mid+count//2)

    # Same pitch for each
    # pitch = 24
    # pitches = tf.repeat(pitch, count)

    one_hot_pitches = tf.one_hot(pitches, hparams['cond_vector_size'], axis=1)
    
    output = gan.generator([seed, one_hot_pitches], training=False)
    samples = tf.reshape(output, [-1, 128, 256])
    x = tf.unstack(samples)

    width = 4
    height = 2
    plt.figure(figsize=(width * 2, height * 4))

    for i, img in enumerate(x):
        plt.subplot(height, width, i+1)
        plt.title(i)
        plt.imshow(x[i])
        #plt.imshow(tf.reverse(x[i], axis=[1]))
        plt.axis('off')

    plt.savefig('output.png')
    #audio = pro.pipeline([
        #pro.denormalize(normalization='specgan', stats=gan_stats),
        #pro.invert_log_melspec(hparams['sample_rate'], hop_length=256, win_length=1028)
    #])(x)

    #output = np.concatenate(list(audio))

    #librosa.output.write_wav('gan_sample.wav', output, hparams['sample_rate'], norm=False)
