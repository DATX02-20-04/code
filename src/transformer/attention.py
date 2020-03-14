import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    attention_logits = qk / tf.math.sqrt(dk)

    if mask is not None:
        attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(attention_logits, axis=-1)

    outputs = tf.matmul(weights, v)

    return outputs, weights
