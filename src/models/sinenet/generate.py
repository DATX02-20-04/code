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
    sinenet = SineNet(hparams, [122, 257])

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        param_net=sinenet.param_net,
    )

    trainer.init_checkpoint(ckpt)

    # ones = tf.ones([1, hparams['channels']])
    # params = tf.tile(tf.concat([ones*10, ones*440], axis=1), [hparams['batch_size'], 1])
    inp = tf.math.abs(tf.random.normal([64, 122, 257]))
    #max_ = tf.math.reduce_max(inp)
    #min_ = tf.math.reduce_min(inp)
    #inp = (inp - min_) / (max_ - min_)
    params = sinenet.param_net(inp, training=False)

    wave = tf.reshape(sinenet.get_wave(params), [-1])

    librosa.output.write_wav('output.wav', wave.numpy(), sr=hparams['sample_rate'], norm=True)
