import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
from data.maestro import maestro_from_files
from models.vae.model import VAE
import tensorflow_datasets as tfds

loss_avg = tf.keras.metrics.Mean()


def on_epoch_start(epoch, step):
    loss_avg.reset_states()

def on_step(epoch, step, loss):
    loss_avg(loss)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_avg.result()}")


def start(hparams):
    # Load nsynth dataset
    # dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)
    dataset = tf.data.Dataset.list_files('/home/big/datasets/maestro-v2.0.0/**/*.wav')
    dataset = pro.pipeline([
        pro.wav(),
        # pro.resample(16000, hparams['sample_rate'], tf.float32),
        pro.normalize(),
        pro.frame(hparams['window_samples'], hparams['window_samples']),
        pro.unbatch(),
        pro.set_channels(1),
        pro.dupe(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(dataset)

    vae = VAE(hparams)

    vae.vae.summary()

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        encoder=vae.encoder,
        decoder=vae.decoder,
        vae=vae.vae,
    )

    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step

    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(vae.train_step)
    trainer.run()
