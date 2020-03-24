import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import os
from models.common.training import Trainer
import data.process as pro
from models.transformer.model import Transformer
import tensorflow_datasets as tfds
from evolve.hparams import HParams


train_loss = tfk.metrics.Mean(name='train_loss')
train_accuracy = tfk.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# This runs at the start of every epoch
def on_epoch_start(epoch, step):
    train_loss.reset_states()
    train_accuracy.reset_states()

# This runs at every step in the training (for each batch in dataset)
def on_step(epoch, step, stats):
    loss, tar_real, predictions = stats
    train_loss(loss)
    train_accuracy(tar_real, predictions)
    if step % 100 == 0:
        print(f"Epoch: {epoch}, Step: {step}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}")

# This runs at the end of every epoch and is used to display metrics
def on_epoch_complete(epoch, step, duration):
    #display.clear_output(wait=True)
    print(f"Epoch: {epoch}, Step: {step}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}, Duration: {duration:.3f}")

def start(hparams):
    hparams = HParams(hparams).generate()
    print(hparams)
    input_vocab_size  = 128+128+100+100
    target_vocab_size = 128+128+100+100

    dataset = tf.data.Dataset.list_files(os.path.join(hparams['dataset_root'] if 'dataset_root' in hparams else './maestro-v2.0.0/', '**/*.midi'))

    dataset_single = pro.pipeline([
        pro.midi(),
        pro.frame(hparams['frame_size'], hparams['frame_size'], True),
        pro.unbatch(),
    ])(dataset)

    def _reshape(inp, tar):
        inp = tf.reshape(inp, [hparams['frame_size']])
        tar = tf.reshape(tar, [hparams['frame_size']])
        return inp, tar


    dataset = pro.pipeline([
        pro.batch(2, True),
        pro.split(2),
        # pro.dupe(),
        pro.map_transform(_reshape),
        pro.cache(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size'], True),
        pro.prefetch(),
    ])(dataset_single)

    dataset_single = dataset_single.as_numpy_iterator()

    transformer = Transformer(input_vocab_size=input_vocab_size,
                              target_vocab_size=target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              hparams=hparams)



    seed = next(dataset_single)

    trainer = Trainer(dataset, hparams)
    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        transformer=transformer,
        optimizer=transformer.optimizer
    )

    trainer.init_checkpoint(ckpt)
    trainer.set_train_step(transformer.train_step)
    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step
    trainer.on_epoch_complete = on_epoch_complete

    trainer.run()
