import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "-1"
import tensorflow as tf
import librosa
import data.process as pro
import matplotlib.pyplot as plt
from models.upscaler.process import load, invert
from models.upscaler.model import Upscaler


def start(hparams):
    dataset, stats = load(hparams)

    valid = stats['examples']//5
    train = stats['examples']-valid

    dataset = dataset.shuffle(1000)
    valid_dataset = dataset.take(valid)
    dataset = dataset.skip(valid).take(train)
    dataset = dataset.batch(8, drop_remainder=True)
    #dataset = dataset.repeat()

    upscaler = Upscaler(hparams, stats)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    upscaler.model.load_weights(checkpoint_path)

    upscaler.model.fit(x=dataset, validation_data=valid_dataset, epochs=100, callbacks=[cp_callback])

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
