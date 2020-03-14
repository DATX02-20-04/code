import tensorflow as tf
from transformer.layers import Encoder, Decoder

tfk = tf.keras
tfkl = tfk.layers

class Transformer(tfk.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tfkl.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        eo = self.encoder(inp, training, enc_padding_mask)

        do, attention_weights = self.decoder(tar, eo, training, look_ahead_mask, dec_padding_mask)

        fo = self.final_layer(do)

        return fo, attention_weights
