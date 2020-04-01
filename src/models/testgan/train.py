import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
from data.nsynth import nsynth_from_tfrecord, instruments, nsynth_to_melspec
from models.testgan.model import TESTGAN
import tensorflow_datasets as tfds

# Define some metrics to be used in the training
gen_loss_avg = tf.keras.metrics.Mean()
disc_loss_avg = tf.keras.metrics.Mean()


def instrument_filter(hparams):
    def _filter(x):
        return tf.reshape(tf.math.equal(x['instrument_family'], instruments[hparams['instrument']]), [])
    return _filter

# This runs at the start of every epoch
def on_epoch_start(epoch, step):
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()

# This runs at every step in the training (for each batch in dataset)
def on_step(epoch, step, stats):
    gen_loss, disc_loss = stats
    gen_loss_avg(gen_loss)
    disc_loss_avg(disc_loss)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}")

# This runs at the end of every epoch and is used to display metrics
def on_epoch_complete(epoch, step, duration):
    #display.clear_output(wait=True)
    print(f"Epoch: {epoch}, Step: {step}, Gen Loss: {gen_loss_avg.result()}, Disc Loss: {disc_loss_avg.result()}, Duration: {duration} s")

def start(hparams):
    # Load nsynth dataset from tfds
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    dataset = nsynth_to_melspec(dataset, hparams)

    # Determine shape of the spectograms in the dataset
    spec_shape = None
    for e in dataset.take(1):
        spec_shape = e.shape
        print(f'Spectogram shape: {spec_shape}')

    # Make sure we got a shape before continuing
    assert spec_shape is not None, "Could not get spectogram shape"

    # Make sure the dimensions of spectogram is divisible by 4.
    # This is because the generator is going to upscale it's state twice with a factor of 2.
    assert spec_shape[0] % 4 == 0 and spec_shape[1] % 4 == 0, "Spectogram dimensions is not divisible by 4"

    # Create preprocessing pipeline for shuffling and batching
    dataset = pro.pipeline([
        pro.set_channels(1),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(dataset)

    testgan = TESTGAN(spec_shape, hparams)

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        generator=testgan.generator,
        selector=testgan.selector,
        gen_optimizer=testgan.generator_optimizer,
        sel_optimizer=testgan.selector_optimizer,
    )

    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(testgan.train_step)
    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step
    trainer.on_epoch_complete = on_epoch_complete

    trainer.run()
