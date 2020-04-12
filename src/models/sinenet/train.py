import tensorflow as tf
import data.process as pro
from models.common.training import Trainer
import tensorflow_datasets as tfds
from models.sinenet.model import SineNet
import tensorflow_datasets as tfds
import librosa

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

    stft_dataset = pro.pipeline([
        pro.extract('audio'),
        pro.normalize(),
        pro.stft(frame_length=2048, frame_step=512, fft_length=126),
        pro.abs(),
        # pro.map_transform(lambda x: tf.reduce_max(x, axis=0)),
    ])(dataset)

    shape = None
    for e in stft_dataset.take(1):
        shape = e.shape

    print("FT shape:", shape)
    sinenet = SineNet(hparams, shape)

    x_dataset = pro.pipeline([
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(stft_dataset)

    y_dataset = pro.pipeline([
        pro.extract('audio'),
        pro.normalize(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
        pro.prefetch()
    ])(dataset)

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    dataset = pro.pipeline([
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size']),
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
