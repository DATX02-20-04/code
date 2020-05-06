import tensorflow as tf
import librosa
import os
import data.process as pro
import matplotlib.pyplot as plt
from models.upscaler.process import load, invert
from models.upscaler.model import Upscaler


def start(hparams):
    dataset, stats = load(hparams)

    dataset = dataset.batch(4, drop_remainder=True)
    dataset = dataset.repeat()

    upscaler = Upscaler(hparams, stats)

    ckpt = tf.train.Checkpoint(
        upscaler=upscaler,
        optimizer=upscaler.optimizer,
    )

    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(hparams['save_dir'], 'ckpts', hparams['name']),
                                         max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    upscaler.model.fit(x=dataset.as_numpy_iterator(), epochs=2)

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
