import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import preprocess as pre
import time
from transformer.model import Transformer
from transformer.optimizer import TransformerLRSchedule
from transformer.mask import create_padding_mask, create_look_ahead_mask

tfk = tf.keras

hparams = {
    'epochs': 20,
    'num_layers': 1,
    'd_model': 329,
    'dff': 128,
    'num_heads': 7,
    'input_vocab_size': 16,
    'target_vocab_size': 16,
    'batch_size': 32,
    'buffer_size': 128,
    'dropout_rate': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.98,
    'epsilon': 1e-9
}

dataset = tf.data.Dataset.list_files('/home/big/datasets/maestro-v2.0.0/**/*.midi')

dataset = pre.pipeline([
    pre.midi(),
    pre.unbatch(),
    pre.batch(16, drop_remainder=True),
    pre.dupe(),
    pre.shuffle(hparams['buffer_size']),
    pre.batch(hparams['batch_size']),
    pre.prefetch(),
])(dataset)

dataset = dataset.take(100).cache().repeat()

transformer = Transformer(num_layers=hparams['num_layers'],
                          d_model=hparams['d_model'],
                          num_heads=hparams['num_heads'],
                          dff=hparams['dff'],
                          input_vocab_size=hparams['input_vocab_size'],
                          target_vocab_size=hparams['target_vocab_size'],
                          pe_input=hparams['input_vocab_size'],
                          pe_target=hparams['target_vocab_size'],
                          rate=hparams['dropout_rate'])

optimizer = tfk.optimizers.Adam(TransformerLRSchedule(hparams['d_model']),
                                beta_1=hparams['beta_1'],
                                beta_2=hparams['beta_2'],
                                epsilon=hparams['epsilon'])

loss_obj = tfk.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_obj(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

train_loss = tfk.metrics.Mean(name='train_loss')
train_accuracy = tfk.metrics.SparseCategoricalAccuracy(name='train_accuracy')

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tar.shape[1], tar.shape[2])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(hparams['epochs']):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    batch = 0
    for inp, tar in dataset:
        print(inp.shape)
        train_step(inp, tar)

        batch += 1
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
