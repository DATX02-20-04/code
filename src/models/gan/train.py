import tensorflow as tf
import json
import numpy as np
import functools
import data.process as pro
from models.common.training import Trainer
from data.nsynth import nsynth_from_tfrecord, nsynth_to_melspec, nsynth_to_cqt_inst, nsynth_to_stft_inst
from models.gan.model import GAN
import data.process as pro
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io

# Define some metrics to be used in the training
gen_loss_avg = tf.keras.metrics.Mean()
disc_loss_avg = tf.keras.metrics.Mean()


# This runs at the start of every epoch
def on_epoch_start(epoch, step, tsw):
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()


# This runs at every step in the training (for each batch in dataset)
def on_step(epoch, step, stats, tsw):
    gen_loss, disc_loss = stats
    gen_loss_avg(gen_loss)
    disc_loss_avg(disc_loss)
    with tsw.as_default():
        tf.summary.scalar('gen_loss', gen_loss_avg.result(), step=step)
        tf.summary.scalar('disc_loss', disc_loss_avg.result(), step=step)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}")

def start(hparams):

    # Load nsynth dataset from tfds
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    gan_stats = calculate_dataset_stats(hparams, dataset)
    # gan_stats = np.load('gan_stats.npz')

    dataset = nsynth_to_stft_inst(dataset, hparams, gan_stats)


    # Determine shape of the spectograms in the dataset
    spec_shape = None
    for x in dataset.take(1):
        e = x['audio']
        cond = x['pitch']
        spec_shape = e.shape
        print(cond)
        print(f'Spectogram shape: {spec_shape}')

    # Make sure we got a shape before continuing
    assert spec_shape is not None, "Could not get spectogram shape"

    # Make sure the dimensions of spectogram is divisible by 4.
    # This is because the generator is going to upscale it's state twice with a factor of 2.
    assert spec_shape[0] % 4 == 0 and spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

    #dataset = pro.index_map('audio', pro.reshape([*spec_shape, 1]))(dataset)

    # Create preprocessing pipeline for shuffling and batching
    dataset = pro.pipeline([
        pro.cache(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(dataset)

    gan = GAN(spec_shape, hparams)
    gan.discriminator.summary()
    gan.generator.summary()


    # This runs at the end of every epoch and is used to display metrics
    def on_epoch_complete(epoch, step, duration, tsw):
        #display.clear_output(wait=True)
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

        magphase = gan.generator([seed, one_hot_pitches], training=False)
        mag = tf.convert_to_tensor(magphase.numpy()[:,:,:,0])
        phase = tf.convert_to_tensor(magphase.numpy()[:,:,:,1])

        mag_samples = tf.reshape(mag, [count, 256, 128])
        phase_samples = tf.reshape(phase, [count, 256, 128])

        img = tf.unstack(mag_samples)
        img = tf.reverse(tf.concat(img, axis=1), axis=[0])

        plt.figure(0, figsize=(count * 2, 4 * 2), clear=True)
        plt.subplot(2,1,1)
        plt.title('Magnitude')
        plt.axis('off')
        plt.imshow(img)

        img = tf.unstack(phase_samples)
        img = tf.reverse(tf.concat(img, axis=1), axis=[0])

        plt.subplot(2,1,2)
        plt.title('Phase')
        plt.axis('off')
        plt.imshow(img)

        buf = io.BytesIO()
        plt.savefig(buf,  format='png')
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Convert to audio
        #audio = pro.pipeline([
            #pro.denormalize(normalization='specgan', stats=gan_stats),
            #pro.inverse_cqt_spec(hparams['sample_rate'])
            #])(magphase)

        #audio = next(audio).reshape([1, -1, 1])
        #audio = next(audio)
        
        # Convert to tf audio
        with tsw.as_default():
            tf.summary.image(f'Spectrogram', image, step=step)
            #tf.summary.audio(f'Audio', audio, hparams['sample_rate'], step=step, encoding='wav')
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}, Duration: {duration} s")


    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=gan.generator,
        discriminator=gan.discriminator,
        gen_optimizer=gan.generator_optimizer,
        disc_optimizer=gan.discriminator_optimizer,
    )

    trainer.init_checkpoint(ckpt)
    trainer.init_tensorboard()
    trainer.set_train_step(gan.train_step)
    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step
    trainer.on_epoch_complete = on_epoch_complete

    trainer.run()

def calculate_dataset_stats(hparams, dataset):
    print("Calculating dataset stats...")
    dataset = nsynth_to_stft_inst(dataset, hparams)

    megabatch = dataset.batch(500).as_numpy_iterator()
    x = next(megabatch)

    spec = x['audio'][:,:,:,0]
    print(f'SPECTROGRAM: {spec.shape}')
    s_mean = spec.mean(axis=0)
    s_min = spec.min(axis=0)
    s_max = spec.max(axis=0)
    s_variance = spec.var(axis=0)
    print(f'mean:{s_mean}, min:{s_min}, max:{s_max}, var:{s_variance}')


    phase = x['audio'][:,:,:,1]
    p_mean = phase.mean(axis=0)
    p_min = phase.min(axis=0)
    p_max = phase.max(axis=0)
    p_variance = phase.var(axis=0)


    np.savez('gan_stats.npz',
             s_mean=s_mean,
             s_min=s_min,
             s_max=s_max,
             s_variance=s_variance,
             p_mean=p_mean,
             p_min=p_min,
             p_max=p_max,
             p_variance=p_variance,
    )

    print("Calculating dataset stats, done.")

    return {
        'spec': {
            'mean': s_mean,
            'min': s_min,
            'max': s_max,
            'variance': s_variance,
        },
        'phase': {
            'mean': p_mean,
            'min': p_min,
            'max': p_max,
            'variance': p_variance,
        }
    }
