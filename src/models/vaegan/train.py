import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
from data.maestro import maestro_from_files
from models.vaegan.model import VAEGAN
import tensorflow_datasets as tfds

vae_loss_avg = tf.keras.metrics.Mean()
critic_loss_avg = tf.keras.metrics.Mean()
similarity_loss_avg = tf.keras.metrics.Mean()


def on_epoch_start(epoch, step, tsw):
    vae_loss_avg.reset_states()
    critic_loss_avg.reset_states()
    similarity_loss_avg.reset_states()

def on_step(epoch, step, stats, tsw):
    vae_loss, critic_loss, similarity_loss = stats
    
    vae_loss_avg(vae_loss)
    critic_loss_avg(critic_loss)
    similarity_loss_avg(similarity_loss)

    with tsw.as_default():
        tf.summary.scalar('vae_loss', vae_loss_avg.result(), step=step)
        tf.summary.scalar('critic_loss', critic_loss_avg.result(), step=step)
        tf.summary.scalar('similarity_loss', similarity_loss_avg.result(), step=step)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, VAE Loss: {vae_loss_avg.result()}, Critic Loss: {critic_loss_avg.result()}, Similarity Loss: {similarity_loss_avg.result()}")

def start(hparams):
    # Load nsynth dataset
    # dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)
    dataset = tf.data.Dataset.list_files('G:\\Delade enheter\\Cool AI Music\\datasets\\maestro-v2.0.0\\**\\*.wav')
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

    vaegan = VAEGAN(hparams)

    vaegan.vaegan.summary()

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        encoder=vaegan.encoder,
        decoder=vaegan.decoder,
        vaegan=vaegan.vaegan,
        critic=vaegan.critic
    )

    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step

    trainer.init_tensorboard()
    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(vaegan.train_step)
    trainer.run()
