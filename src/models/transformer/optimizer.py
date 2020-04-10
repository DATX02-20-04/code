import tensorflow as tf
import tensorflow.keras as tfk

''' Custom learning rate scheduler for the optimizer (formula follows transformer/CustomLearningRateScheduler.png).
    Source: "Attention Is All You Need" '''

class TransformerLRSchedule(tfk.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerLRSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        a = tf.math.rsqrt(step)
        b = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(a, b)
