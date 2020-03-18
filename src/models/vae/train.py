import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
from data.maestro import maestro_from_files
from models.vae.model import VAE
import tensorflow_datasets as tfds

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def start(hparams):
    # Load nsynth dataset
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)
    dataset = pro.pipeline([
        pro.extract('audio'),
        pro.frame(hparams['window_samples'], hparams['window_samples']),
        pro.unbatch(),
        pro.set_channels(1),
        pro.dupe(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(dataset)

    vae = VAE(hparams)

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        encoder=vae.encoder,
        decoder=vae.decoder,
        vae=vae.vae,
    )

    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(vae.train_step)
    trainer.run()
