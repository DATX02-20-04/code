import tensorflow as tf
import numpy as np
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
import librosa
from models.sinenet.dataset import create_dataset
from data.nsynth import instrument_families_filter

loss_avg = tf.keras.metrics.Mean()

def on_epoch_start(epoch, step, tsw):
    loss_avg.reset_states()

def on_step(epoch, step, loss, tsw):
    loss_avg(loss)
    if step % 5 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_avg.result()}")
    with tsw.as_default():
        tf.summary.scalar('loss', loss_avg.result(), step=step)

def start(hparams):
    dataset = create_dataset(hparams)

    sinenet = SineNet(hparams)

    # sinenet.summary()

    dataset = pro.pipeline([
        pro.cache(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size'], drop_remainder=True),
        pro.prefetch()
    ])(dataset)

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        model=sinenet,
    )

    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step

    trainer.init_tensorboard()
    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(sinenet.train_step)
    trainer.run()
