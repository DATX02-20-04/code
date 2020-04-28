import tensorflow as tf
from models.common.training import Trainer
from models.gan.model import GAN
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import data.process as pro
import numpy as np


def start(hparams):

    y, sr = librosa.load(librosa.util.example_audio_file())
    print(librosa.stft(y).shape)

    gan_stats = np.load('gan_stats.npz')

    gan = GAN((256, 128, 2), hparams)

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)
    count = 1

    # Random seed for each
    seed = tf.random.normal((count, hparams['latent_size']))

    # Same seed for all
    # seed = tf.random.normal((1, hparams['latent_size']))
    # seed = tf.repeat(seed, count, axis=0)

    # Sample pitch from middle
    #mid = hparams['cond_vector_size']//2
    #pitches = range(mid-count//2, mid+count//2)

    # Same pitch for each
    pitch = 24
    pitches = tf.repeat(pitch, count)

    one_hot_pitches = tf.one_hot(pitches, hparams['cond_vector_size'], axis=1)

    output = gan.generator([seed, one_hot_pitches], training=False)

    width = 1
    height = 1

    outer, out_axes = plt.subplots(width, height)

    for i, img in enumerate(output):
        magphase = tf.unstack(img, axis=-1)
        mag_sample = magphase[0]
        phase_sample = magphase[1]

        mp, axes = plt.subplots(1,2)
        magax, phax = axes

        pitch = pitches.numpy()[i]
        mp.suptitle(pitch)

        magax.set_title('Magnitude')
        #magax.axis('off')
        magax.imshow(mag_sample, origin='lower')

        phax.set_title('Phase')
        #phax.axis('off')
        phax.imshow(phase_sample, origin='lower')


    plt.savefig('stft.png')

    # Add the batch dimension
    image = tf.expand_dims(output, 0)


    # Convert to audio
    audio = pro.pipeline([
        pro.denormalize(normalization='specgan_two_channel', stats=gan_stats),
        pro.istft_spec(hop_length=512, win_length=512),
        ])(output)

    print(audio)

    librosa.output.write_wav('stft.wav', audio.numpy(), hparams['sample_rate'], norm=True)
