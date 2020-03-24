import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers as tfkl
from models.transformer.layers import Encoder, Decoder
from models.transformer.mask import create_padding_mask, create_look_ahead_mask
from models.transformer.optimizer import TransformerLRSchedule

# Transformer class. Initialize components of the transformer nerual network architecture
class Transformer(tfk.Model):
    def __init__(self, input_vocab_size, target_vocab_size, pe_input, pe_target, hparams):
        super(Transformer, self).__init__()
        self.hparams = hparams

        # Encoder
        self.encoder = Encoder(hparams['num_layers'],
                               hparams['d_model'],
                               hparams['num_heads'],
                               hparams['dff'],
                               input_vocab_size,
                               pe_input,
                               hparams['dropout_rate'])

        # Decoder
        self.decoder = Decoder(hparams['num_layers'],
                               hparams['d_model'],
                               hparams['num_heads'],
                               hparams['dff'],
                               target_vocab_size,
                               pe_target,
                               hparams['dropout_rate'])

        self.final_layer = tfkl.Dense(target_vocab_size)

        self.loss_obj = tfk.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.optimizer = tfk.optimizers.Adam(TransformerLRSchedule(hparams['d_model']),
                                             beta_1=hparams['beta_1'],
                                             beta_2=hparams['beta_2'],
                                             epsilon=hparams['epsilon'])

    # TODO
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        eo = self.encoder(inp, training, enc_padding_mask)

        do, attention_weights = self.decoder(tar, eo, training, look_ahead_mask, dec_padding_mask)

        fo = self.final_layer(do)

        return fo, attention_weights

    # Instruction for each training step
    @tf.function
    def train_step(self, x):
        inp, tar = x
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.call(inp, tar_inp,
                                       True,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, tar_real, predictions

    # loss function
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self.loss_obj(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_mean(loss)

    # TODO
    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    # TODO
    def evaluate(self, inp_sentence):
        encoder_input = tf.expand_dims(inp_sentence, 0)

        decoder_input = [inp_sentence[0]]
        output = tf.expand_dims(decoder_input, 0)
        output_tot = tf.expand_dims(decoder_input, 0)

        for i in range(self.hparams['frame_size']):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                encoder_input, output)

            predictions, attention_weights = self.call(encoder_input,
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
