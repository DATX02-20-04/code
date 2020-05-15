import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
from models.common.training import Trainer
import data.process as pro
import data.midi as M
from models.transformer.model import Transformer
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def create_run(hparams, logger, span, **kwargs):
    input_vocab_size  = 128+128+128+128
    target_vocab_size = 128+128+128+128

    transformer = Transformer(input_vocab_size=input_vocab_size,
                              target_vocab_size=target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              hparams=hparams)


    trainer = Trainer(None, hparams)

    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        transformer=transformer,
        optimizer=transformer.optimizer
    )

    trainer.init_checkpoint(ckpt)

    gen_iters = kwargs['gen_iters'] if 'gen_iters' in kwargs else 1

    def run(prior):
        output = prior
        outputs = []

        span('start', 'melody_gen')
        for i in range(gen_iters):
            s = f"gen_{i+1}/{gen_iters}"
            span('start', s)
            output, _ = transformer.evaluate(output)
            outputs.append(output)
            output = output + tf.random.uniform(output.shape, -4, 4, dtype=tf.int32)
            span('end', s)
        span('end', 'melody_gen')

        encoded = tf.concat(outputs, 0)
        prior = tf.concat(prior, 0)

        return encoded
