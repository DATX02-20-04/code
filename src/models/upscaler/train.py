import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "-1"
import tensorflow as tf
import librosa
import data.process as pro
import matplotlib.pyplot as plt
from models.upscaler.process import load, invert
from models.upscaler.model import Upscaler
from util import get_plot_image

import datetime
import time

class TFImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, tbw, dataset):
        self.tbw = tbw
        self.step = 0
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        for X, Y in self.dataset.take(1):
            P = self.model.predict(X)

            X = tf.squeeze(X[0])
            Y = tf.squeeze(Y[0])
            P = tf.squeeze(P[0])

            plt.clf()
            fig, axs = plt.subplots(1, 3)
            axs[0].axes.get_xaxis().set_visible(False)
            axs[0].axes.get_yaxis().set_visible(False)
            axs[0].imshow(tf.transpose(X, [1, 0]), origin='bottom')
            axs[1].axes.get_xaxis().set_visible(False)
            axs[1].axes.get_yaxis().set_visible(False)
            axs[1].imshow(tf.transpose(Y, [1, 0]), origin='bottom')
            axs[2].axes.get_xaxis().set_visible(False)
            axs[2].axes.get_yaxis().set_visible(False)
            axs[2].imshow(tf.transpose(P, [1, 0]), origin='bottom')
            plt.tight_layout()
            img = get_plot_image()

            with self.tbw.as_default():
                tf.summary.image(f'XYP', img, step=self.step)

            self.step += 1



def start(hparams):
    dataset, stats = load(hparams)

    tb_writer = init_tensorboard(hparams)

    valid = stats['examples']//10
    train = stats['examples']-valid

    dataset = dataset.map(lambda x, y: (tf.reshape(x, [128, 256]), tf.reshape(y, [128, 1024])))

    valid_dataset = dataset.take(valid)
    valid_dataset = dataset.batch(8, drop_remainder=True)
    dataset = dataset.skip(valid).take(train)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(8, drop_remainder=True)
    #dataset = dataset.repeat()

    upscaler = Upscaler(hparams, stats)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./logs/{hparams['name']}/{current_time}/train/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    tf_image_cb = TFImageCallback(tb_writer, dataset)

    try:
        upscaler.model.load_weights(checkpoint_path)
    except:
        print("Initializing from scratch.")

    upscaler.model.fit(x=dataset, validation_data=valid_dataset, epochs=100, callbacks=[cp_callback, tensorboard_callback, tf_image_cb])

    # plot_magphase(hparams, gen, f'generated_magphase_block{i:02d}')
    # invert_magphase(hparams, stats, gen, f'generated_magphase_block{i:02d}')


def plot_magphase(hparams, magphase, name, pitch=None):
    assert len(magphase.shape) == 4, "Magphase needs to be in the form (batch, width, height, channels)"
    count = magphase.shape[0]
    fig, axs = plt.subplots(1, 2*count)
    for i in range(count):
        mag, phase = tf.unstack(magphase[i], axis=-1)
        if pitch is not None:
            plt.suptitle(f"Pitch: {tf.argmax(pitch)}")
        axs[0+i*2].set_title("Mag")
        axs[0+i*2].axes.get_xaxis().set_visible(False)
        axs[0+i*2].axes.get_yaxis().set_visible(False)
        axs[0+i*2].imshow(tf.transpose(mag, [1, 0]))
        axs[1+i*2].set_title("Pha")
        axs[1+i*2].axes.get_xaxis().set_visible(False)
        axs[1+i*2].axes.get_yaxis().set_visible(False)
        axs[1+i*2].imshow(tf.transpose(phase, [1, 0]))

    plt.tight_layout()
    plt.savefig(f'{name}.png')

def invert_magphase(hparams, stats, magphase, name):
    assert len(magphase.shape) == 4, "Magphase needs to be in the form (batch, width, height, channels)"
    count = magphase.shape[0]
    audio = []
    for i in range(count):
        mag, phase = tf.unstack(magphase[i], axis=-1)
        audio.append(invert(hparams, stats)((mag, phase)))
    audio = tf.concat(audio, axis=0)
    librosa.output.write_wav(f'{name}.wav', audio.numpy(), sr=hparams['sample_rate'])

def init_tensorboard(hparams):
    # Tensorfboard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f"./logs/{hparams['name']}/{current_time}/train/"
    return tf.summary.create_file_writer(train_log_dir)
