import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
from models.common.training import Trainer
import data.process as pro
import data.midi as M
from models.transformer.model import Transformer
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def generate(hparams, prior_index, gen_iters=None):
    input_vocab_size  = 128+128+128+128
    target_vocab_size = 128+128+128+128

    dataset = tf.data.Dataset.list_files(f"{hparams['dataset_root']}/maestro-v2.0.0/**/*.midi")

    dataset_single = pro.pipeline([
        pro.midi(),
        pro.frame(hparams['frame_size'], hparams['frame_hop_len'], True),
        pro.unbatch(),
    ])(dataset).skip(prior_index).as_numpy_iterator()

    transformer = Transformer(input_vocab_size=input_vocab_size,
                              target_vocab_size=target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              hparams=hparams)


    trainer = Trainer(dataset, hparams)
    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        transformer=transformer,
        optimizer=transformer.optimizer
    )

    trainer.init_checkpoint(ckpt)

    return generate_from_model(hparams, transformer, dataset_single, gen_iters)

def generate_from_model(hparams, transformer, priors, gen_iters=None):
    if gen_iters is None:
        gen_iters = hparams['gen_iters'] if 'gen_iters' in hparams else 1
    outputs = []
    prior_buf = []
    prior = np.array(next(priors))
    output = prior
    for i in range(gen_iters):
        print(f"Generating {i+1}/{gen_iters} sample...")
        output, _ = transformer.evaluate(output)
        outputs.append(output)
        output = output + tf.random.uniform(output.shape, -4, 4, dtype=tf.int32)

    encoded = tf.concat(outputs, 0)
    prior = tf.concat(prior, 0)
    return encoded, prior

def start(hparams):
    i = 1
    encoded, prior = generate(hparams)
    decoded = pro.decode_midi()(encoded)
    with open('gen_transformer_{}.midi'.format(i), 'wb') as f:
        M.write_midi(f, decoded)

    M.display_midi(decoded)
    plt.savefig('gen_transformer_{}.png'.format(i))

    print('Generated sample')
