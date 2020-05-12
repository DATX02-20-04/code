import tensorflow as tf
import librosa
import os
import io
import data.process as pro
import matplotlib.pyplot as plt
import datetime
import time
from models.new_gan.process import load, invert
from models.new_gan.model import GAN
from util import get_plot_image


def start(hparams):
    dataset, stats = load(hparams)

    def resize(image, down_scale):
        return tf.squeeze(tf.image.resize(tf.reshape(image, [1, 128, 256, 1]), [128//down_scale, 256//down_scale]))

    gan = GAN(hparams, stats)
    block = tf.Variable(0)
    step = tf.Variable(0)
    pitch_start = 0
    pitch_end = hparams['pitches']
    step_size = 1
    seed_pitches = tf.range(pitch_start, pitch_start+pitch_end, step_size)
    seed = tf.Variable(tf.random.normal([seed_pitches.shape[0], hparams['latent_dim']]))
    seed_pitches = tf.one_hot(seed_pitches, hparams['pitches'], axis=1)

    tsw = init_tensorboard(hparams)

    ckpt = tf.train.Checkpoint(
        gan=gan,
        seed=seed,
        generator_optimizer=gan.generator_optimizer,
        discriminator_optimizer=gan.discriminator_optimizer,
        block=block,
        step=step,
    )

    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(hparams['save_dir'], 'ckpts', hparams['name']),
                                         max_to_keep=3)
    dataset = dataset.shuffle(hparams['buffer_size'])


    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {} block {}".format(manager.latest_checkpoint, block.numpy()))
    else:
        print("Initializing from scratch.")

    if block == 0:
        # Get the first models to train
        g_init, d_init = gan.get_initial_models()

        # Create the smallest scaled dataset to train the first
        scaled_dataset = pro.pipeline([
            pro.map_transform(lambda magphase, pitch: (resize(magphase, 2**(hparams['n_blocks']-1)), pitch)),
            pro.cache(),
        ])(dataset)

        block.assign_add(1)
        gan.train_epochs(g_init, d_init, scaled_dataset, hparams['epochs'][0], hparams['batch_sizes'][0], block, step, tsw=tsw)

        manager.save()

        X_fake = g_init([seed, seed_pitches], training=False)
        plot_magphase(hparams, X_fake, step, block, tsw=tsw, pitch=seed_pitches)

    for i in range(block.numpy(), hparams['n_blocks']):
        down_scale = 2**(hparams['n_blocks']-i-1)
        batch_size = hparams['batch_sizes'][i]
        epochs = hparams['epochs'][i]

        scaled_dataset = pro.pipeline([
            pro.map_transform(lambda magphase, pitch: (resize(magphase, down_scale), pitch)),
            pro.cache(),
        ])(dataset)

        [g_normal, g_fadein] = gan.generators[i]
        [d_normal, d_fadein] = gan.discriminators[i]

        block.assign_add(1)

        print("\nFading in next...")
        gan.train_epochs(g_fadein, d_fadein, scaled_dataset, epochs, batch_size, block, step, fade=True)

        print("\nNormal training...")
        gan.train_epochs(g_normal, d_normal, scaled_dataset, epochs, batch_size, block, step, tsw=tsw)

        manager.save()

        X_fake = g_normal([seed, seed_pitches], training=False)
        plot_magphase(hparams, X_fake, step, block, tsw=tsw, pitch=seed_pitches)

    final_epochs = 100
    batch_size = 32
    last = hparams['n_blocks']-1
    [g_normal, g_fadein] = gan.generators[last]
    [d_normal, d_fadein] = gan.discriminators[last]

    block.assign_add(1)
    for i in range(1, final_epochs+1):
        print(f"\nFinal training {i}/{final_epochs}...")
        gan.train_epochs(g_normal, d_normal, dataset, 1, batch_size, block, step, tsw=tsw)

        manager.save()
        X_fake = g_normal([seed, seed_pitches], training=False)
        plot_magphase(hparams, X_fake, step, block, tsw=tsw, pitch=seed_pitches)


def plot_magphase(hparams, magphase, step, block=None, tsw=None, pitch=None):
    assert len(magphase.shape) == 4, "Magphase needs to be in the form (batch, width, height, channels)"
    mag = tf.unstack(magphase, axis=0)
    mag = tf.concat(mag, axis=0)
    mag = tf.squeeze(mag)
    aspect = mag.shape[0]/mag.shape[1]
    fig = plt.figure(figsize=(aspect*2, 2))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.imshow(tf.transpose(mag, [1, 0]), origin='bottom')

    plt.tight_layout()

    if tsw is not None:
        tb_block = block.numpy() if block is not None else None
        tb_step = step.numpy()
        img = get_plot_image()
        with tf.name_scope(f'Block: {tb_block}' if tb_block is not None else 'Final Training') as scope:
            with tsw.as_default():
                tf.summary.image(f'Spectrogram', img, step=tb_step)


def invert_magphase(hparams, stats, magphase, name):
    assert len(magphase.shape) == 4, "Magphase needs to be in the form (batch, width, height, channels)"
    count = magphase.shape[0]
    audio = []
    for i in range(count):
        mag, phase = tf.unstack(magphase[i], axis=-1)
        audio.append(invert(hparams, stats)((mag, phase)))
    audio = tf.concat(audio, axis=0)
    librosa.output.write_wav(f'{name}.wav', audio.numpy(), sr=hparams['sample_rate'])


def init_tensorboard(hparams):
    # Tensorfboard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"./logs/{hparams['name']}/{current_time}/train/"
    return tf.summary.create_file_writer(train_log_dir)

