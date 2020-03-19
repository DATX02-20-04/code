import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers as tfkl
from models.transformer.positional import positional_encoding

class ScaledAttention(tfkl.Layer):
    def __init__(self):
        super(ScaledAttention, self).__init__()

    def call(self, q, k, v, mask):
        qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        attention_logits = qk / tf.math.sqrt(dk)

        if mask is not None:
            attention_logits += (mask * -1e9)

        weights = tf.nn.softmax(attention_logits, axis=-1)

        outputs = tf.matmul(weights, v)

        return outputs, weights


class MultiHeadAttention(tfkl.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model % self.num_heads == 0, "d_model is not divisible by num_heads"

        self.depth = self.d_model // self.num_heads

        self.wv = tfkl.Dense(d_model)
        self.wk = tfkl.Dense(d_model)
        self.wq = tfkl.Dense(d_model)

        self.scaled_attention = ScaledAttention()

        self.dense = tfkl.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        v = self.wv(v)
        k = self.wk(k)
        q = self.wq(q)

        v = self.split_heads(v, batch_size)
        k = self.split_heads(k, batch_size)
        q = self.split_heads(q, batch_size)

        attention, attention_weights = self.scaled_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, [batch_size, -1, self.d_model])
        output = self.dense(attention)

        return output, attention_weights


class PointWiseFF(tfkl.Layer):
    def __init__(self, d_model, dff):
        super(PointWiseFF, self).__init__()
        self.d1 = tfkl.Dense(dff, activation='relu')
        self.d2 = tfkl.Dense(d_model)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x


class EncoderLayer(tfkl.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.pwff = PointWiseFF(d_model, dff)

        self.lnorm1 = tfkl.LayerNormalization(epsilon=1e-6)
        self.lnorm2 = tfkl.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tfkl.Dropout(rate)
        self.dropout2 = tfkl.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.lnorm1(x + attn_output)

        ffn_output = self.pwff(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.lnorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tfkl.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.pwff = PointWiseFF(d_model, dff)

        self.lnorm1 = tfkl.LayerNormalization(epsilon=1e-6)
        self.lnorm2 = tfkl.LayerNormalization(epsilon=1e-6)
        self.lnorm3 = tfkl.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tfkl.Dropout(rate)
        self.dropout2 = tfkl.Dropout(rate)
        self.dropout3 = tfkl.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.lnorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.lnorm2(attn2 + out1)

        ffn_output = self.pwff(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.lnorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tfkl.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_pos_enc, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tfkl.Embedding(input_vocab_size, d_model)
        self.pos_enc = positional_encoding(max_pos_enc, self.d_model)

        self.layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tfkl.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_enc[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.layers[i](x, training, mask)

        return x

class Decoder(tfkl.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_pos_enc, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tfkl.Embedding(target_vocab_size, d_model)
        self.pos_enc = positional_encoding(max_pos_enc, d_model)

        self.layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tfkl.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_enc[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['dec_l{}_b1'.format(i+1)] = block1
            attention_weights['dec_l{}_b2'.format(i+1)] = block2

        return x, attention_weights
