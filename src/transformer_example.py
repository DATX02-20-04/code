import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import preprocess as pre
import time
from transformer.model import Transformer
from transformer.optimizer import TransformerLRSchedule
from transformer.mask import create_padding_mask, create_look_ahead_mask
import tensorflow_datasets as tfds

tfk = tf.keras

hparams = {
    'epochs': 20,
    'num_layers': 4,
    'd_model': 128,
    'dff': 512,
    'num_heads': 8,
    'frame_size': 256,
    'input_vocab_size': 128+128+100+100,
    'target_vocab_size': 128+128+100+100,
    'batch_size': 32,
    'buffer_size': 10000,
    'dropout_rate': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.98,
    'epsilon': 1e-9
}

dataset = tf.data.Dataset.list_files('/home/big/datasets/maestro-v2.0.0/**/*.midi')

dataset_single = pre.pipeline([
    pre.midi(),
    pre.frame(hparams['frame_size'], hparams['frame_size'], True),
    pre.unbatch(),
])(dataset)


dataset = pre.pipeline([
    pre.batch(2, True),
    pre.split(2),
    pre.cache(),
    pre.shuffle(hparams['buffer_size']),
    pre.batch(hparams['batch_size'], True),
    pre.prefetch(),
])(dataset_single)

dataset_single = dataset_single.as_numpy_iterator()

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

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
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


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
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


def evaluate(inp_sentence):
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [64]
    output = tf.expand_dims(decoder_input, 0)
    output_tot = tf.expand_dims(decoder_input, 0)

    for i in range(hparams['frame_size']):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        predictions, attention_weights = transformer(encoder_input,
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

        predictions = predictions[: ,-1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # if predicted_id == hparams[]+1:
        #     return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

seed = next(dataset_single)
def generate(epoch):
    output = seed
    print(output)
    outputs = []
    for i in range(4):
        output, _ = evaluate(output)
        outputs.append(output)
        print(output)
    decoded = next(pre.decode_midi()(iter([tf.concat(outputs, 0)])))
    decoded.save('gen_e{}.midi'.format(epoch))


# generate(0)

for epoch in range(hparams['epochs']):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    batch = 0
    for inp, tar in dataset:
        inp = tf.reshape(inp, [hparams['batch_size'], hparams['frame_size']])
        tar = tf.reshape(tar, [hparams['batch_size'], hparams['frame_size']])
        train_step(inp, tar)

        batch += 1
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
    generate(epoch)

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


