import tensorflow as tf
from models.common.training import Trainer
import data.process as pro
import tensorflow_datasets as tfds
from models.gan.model import GAN
from data.nsynth import nsynth_from_tfrecord, nsynth_to_melspec, nsynth_to_cqt_inst, nsynth_to_stft_inst
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import data.process as pro
import numpy as np
from models.gan.train import calculate_dataset_stats


def start(hparams):

    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    # gan_stats = calculate_dataset_stats(hparams, dataset)
    gan_stats = np.load('gan_stats.npz')

    use_norm = True

    dataset = nsynth_to_stft_inst(dataset, hparams, gan_stats if use_norm else None)
    dataset = pro.pipeline([
        pro.extract('audio'),
    ])(dataset).skip(100).as_numpy_iterator()
# gan = GAN((256, 128, 2), hparams)

    # trainer = Trainer(None, hparams)

    # ckpt = tf.train.Checkpoint(
    #     step=trainer.step,
    #     generator=gan.generator,
    #     discriminator=gan.discriminator,
    #     gen_optimizer=gan.generator_optimizer,
    #     disc_optimizer=gan.discriminator_optimizer,
    # )

    # trainer.init_checkpoint(ckpt)
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

    # output = gan.generator([seed, one_hot_pitches], training=False)
    output = next(dataset)
    print("OUTPUT", output.shape)

    width = 1
    height = 1

    outer, out_axes = plt.subplots(width, height)

    # for i, img in enumerate(output):
    magphase = tf.unstack(output, axis=-1)
    mag_sample = magphase[0]
    phase_sample = magphase[1]

    mp, axes = plt.subplots(1,2)
    magax, phax = axes

    # pitch = pitches.numpy()[0]
    # mp.suptitle(pitch)

    magax.set_title('Magnitude')
    #magax.axis('off')
    magax.imshow(mag_sample, origin='lower')

    phax.set_title('Phase')
    #phax.axis('off')
    phax.imshow(phase_sample, origin='lower')

    name = f'stft_norm{use_norm}'

    plt.savefig(f'{name}.png')

    # Convert to audio
    output = tf.constant(output)
    print("OUTPUT", output)
    if use_norm:
        output = pro.denormalize(normalization='specgan_two_channel', stats=gan_stats)(output)

    audio = pro.pipeline([
        pro.istft_spec(hop_length=512, win_length=512),
    ])(output)

    librosa.output.write_wav(f'{name}.wav', audio.numpy(), hparams['sample_rate'], norm=False)
