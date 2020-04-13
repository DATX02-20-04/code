import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
from data.nsynth import instrument_families_filter
import librosa

def note(i):
    return 440*2**(i/12)

def start(hparams):
    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    sinenet = SineNet(hparams, [61])

    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        param_net=sinenet.param_net,
    )

    trainer.init_checkpoint(ckpt)

    # ones = tf.ones([1, hparams['channels']])
    # params = tf.tile(tf.concat([ones*10, ones*440], axis=1), [hparams['batch_size'], 1])
    start = 0
    end = 60
    inp = tf.one_hot(tf.range(start, end), 61)
    print(inp.shape)
    #max_ = tf.math.reduce_max(inp)
    #min_ = tf.math.reduce_min(inp)
    #inp = (inp - min_) / (max_ - min_)
    params = sinenet.param_net(inp, training=False)

    wave = tf.reshape(sinenet.get_wave(params), [-1])

    librosa.output.write_wav('output.wav', wave.numpy(), sr=hparams['sample_rate'], norm=True)
