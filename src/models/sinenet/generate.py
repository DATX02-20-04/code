import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
import librosa

def note(i):
    return 440*2**(i/12)

def start(hparams):
    sinenet = SineNet(hparams, [64])

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        param_net=sinenet.param_net,
    )

    trainer.init_checkpoint(ckpt)

    # ones = tf.ones([1, hparams['channels']])
    # params = tf.tile(tf.concat([ones*10, ones*440], axis=1), [hparams['batch_size'], 1])
    params = sinenet.param_net(tf.random.normal([hparams['batch_size'], 64]), training=False)

    wave = tf.reshape(sinenet.get_wave(params), [-1])

    librosa.output.write_wav('output.wav', wave.numpy(), sr=hparams['sample_rate'])
