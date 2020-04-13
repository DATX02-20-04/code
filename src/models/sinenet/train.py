import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
import librosa
from data.nsynth import instrument_families_filter

loss_avg = tf.keras.metrics.Mean()

def on_epoch_start(epoch, step, tsw):
    loss_avg.reset_states()

def on_step(epoch, step, loss, tsw):
    loss_avg(loss)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_avg.result()}")
    with tsw.as_default():
        tf.summary.scalar('loss', loss_avg.result(), step=step)

def note(i):
    return 440*2**(i/12)

def start(hparams):

    # ones = tf.ones([hparams['channels']])
    # wave = sinenet.get_wave(tf.concat([ones*10, ones*440], axis=0))

    # librosa.output.write_wav('output.wav', wave.numpy(), sr=hparams['sample_rate'])

    dataset = tfds.load('nsynth/gansynth_subset', split='train', shuffle_files=True)

    if 'instrument' in hparams and hparams['instrument'] is not None:
        instrument = hparams['instrument']
        if 'family' in instrument and instrument['family'] is not None:
            print("FILTER ", instrument['family'])
            dataset = pro.filter(instrument_families_filter(instrument['family']))(dataset)

    y_dataset = pro.pipeline([
        pro.extract('audio'),
        pro.normalize(),
    ])(dataset)

    x_dataset = pro.pipeline([
        pro.extract('pitch'),
        pro.map_transform(lambda x: x-24),
        pro.one_hot(61),
        #pro.normalize(),
        #pro.stft(frame_length=2048, frame_step=512, fft_length=512),
        #pro.abs(),
        #pro.map_transform(lambda x: tf.reduce_max(x, axis=0)),
    ])(dataset)

    shape = None
    for e in x_dataset.take(1):
        print(e)
        shape = e.shape

    print("FT shape:", shape)
    sinenet = SineNet(hparams, shape)

    sinenet.param_net.summary()

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    dataset = pro.pipeline([
        pro.cache(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size'], drop_remainder=True),
        pro.prefetch()
    ])(dataset)

    trainer = Trainer(dataset, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        param_net=sinenet.param_net,
    )

    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step

    trainer.init_tensorboard()
    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(sinenet.train_step)
    trainer.run()
