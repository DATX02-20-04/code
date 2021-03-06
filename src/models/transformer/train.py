import tensorflow as tf
import gc
import io
import tensorflow.keras as tfk
import numpy as np
import data.midi as M
import os
from models.common.training import Trainer
import matplotlib.pyplot as plt
import data.process as pro
from models.transformer.model import Transformer
from models.transformer.generate import generate_from_model
import tensorflow_datasets as tfds
from evolve.hparams import HParams
from evolve.pool import Pool
import util

train_loss = tfk.metrics.Mean(name='train_loss')
train_accuracy = tfk.metrics.SparseCategoricalAccuracy(name='train_accuracy')

input_vocab_size  = 128+128+128+128
target_vocab_size = 128+128+128+128

# This runs at the start of every epoch
def on_epoch_start(epoch, step, tsw):
    train_loss.reset_states()
    train_accuracy.reset_states()


# This runs at the end of every epoch and is used to display metrics
def on_epoch_complete(epoch, step, duration, tsw):

    #display.clear_output(wait=True)
    print(f"Epoch: {epoch}, Step: {step}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}, Duration: {duration:.3f}")

def create_model(hp):
    print(hp)
    return Transformer(input_vocab_size=input_vocab_size,
                       target_vocab_size=target_vocab_size,
                       pe_input=input_vocab_size,
                       pe_target=target_vocab_size,
                       hparams=hp)

def evaluate(dataset, hparams):
    def _eval(model):
        trainer = Trainer(dataset, hparams)

        trainer.set_train_step(model.train_step)
        trainer.on_epoch_start = on_epoch_start
        trainer.on_step = on_step
        trainer.on_epoch_complete = on_epoch_complete


        stats = trainer.run()
        if stats is not None:
            loss, _, _ = stats
            return 1 / tf.reduce_mean(loss).numpy()
        else:
            return 0
    return _eval

def start(hparams):
    gc.collect()

    dataset = tf.data.Dataset.list_files('/home/big/datasets/maestro-v2.0.0/**/*.midi')

    dataset_single = pro.pipeline([
        pro.midi(),
        pro.frame(hparams['frame_size']*2, hparams['frame_hop_len'], True),
        pro.unbatch(),
    ])(dataset)

    def _reshape(inp, tar):
        inp = tf.reshape(inp, [hparams['frame_size']])
        tar = tf.reshape(tar, [hparams['frame_size']])
        return inp, tar


    dataset = pro.pipeline([
        pro.split(2),
        #pro.batch(2, True),
        # pro.dupe(),
        pro.map_transform(_reshape),
        pro.cache(),
        pro.shuffle(hparams['buffer_size']),
        pro.batch(hparams['batch_size'], True),
        pro.prefetch(),
    ])(dataset_single)


    dataset_single = pro.shuffle(hparams['buffer_size']//4)(dataset_single)
    dataset_single = dataset_single.as_numpy_iterator()

    transformer = Transformer(input_vocab_size=input_vocab_size,
                              target_vocab_size=target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              hparams=hparams)

    # pop_size = 10
    # generations = 100
    # mutation_rate = 0.2
    # pool.populate(pop_size, 1)

    # for generation in range(generations):
    #     pool.evaluate(evaluate(dataset, hparams))
    #     print(f"--- GENERATION: {generation} ---")
    #     print("BEST:", pool.best, pool.fitness)
    #     pool.select(pop_size)
    #     pool.populate(pop_size, mutation_rate)
    #


    image_save_step = hparams['image_save_step'] if 'image_save_step' in hparams else 2000

    def generate_image(step, tsw):
        print("Generating sample...")
        encoded, seed = generate_from_model(hparams, transformer, dataset_single)
        print("Generating sample done.")

        print("Decoding midi...")
        decoded_seed = pro.decode_midi()(seed)
        decoded = pro.decode_midi()(encoded)
        print("Decoding midi done.")

        print("Saving midi...")
        with open(f'gen_transformer_{step}.midi', 'wb') as f:
            M.write_midi(f, decoded)
        with open(f'prior_transformer_{step}.midi', 'wb') as f:
            M.write_midi(f, decoded_seed)
        print("Saving midi done.")

        print("Plotting midi...")
        plt.title('Prior')
        M.display_midi(decoded_seed)
        image_seed = util.get_plot_image()
        plt.clf()

        plt.title('Generated')
        M.display_midi(decoded)
        image = util.get_plot_image()
        plt.clf()

        image_conc = tf.concat([image_seed, image], axis=1)
        print("Plotting done.")

        with tsw.as_default():
            tf.summary.image(f'image', image_conc, step=step)
        print("Complete.")


    # This runs at every step in the training (for each batch in dataset)
    def on_step(epoch, step, stats, tsw):
        loss, tar_real, predictions = stats
        train_loss(loss)
        train_accuracy(tar_real, predictions)
        if step % 100 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}")
        if step % image_save_step == 0:
            generate_image(step, tsw)

        with tsw.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=step)



    trainer = Trainer(dataset, hparams)
    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        transformer=transformer,
        optimizer=transformer.optimizer
    )

    trainer.init_checkpoint(ckpt)
    trainer.init_tensorboard()
    trainer.set_train_step(transformer.train_step)
    trainer.on_epoch_start = on_epoch_start
    trainer.on_step = on_step
    trainer.on_epoch_complete = on_epoch_complete

    #generate_image(trainer.step.numpy(), trainer.train_summary_writer)

    trainer.run()
