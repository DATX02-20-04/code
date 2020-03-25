import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
from models.common.training import Trainer
import data.process as pro
from models.transformer.model import Transformer
import tensorflow_datasets as tfds

''' #TODO '''

def generate(transformer, epoch, seed):
    ''' generate
        #TODO '''
    output = seed
    print(output)
    outputs = []
    for i in range(1):
        output, _ = transformer.evaluate(output)
        outputs.append(output)
        print(output)
    decoded = next(pro.decode_midi()(iter([tf.concat(outputs, 0)])))
    decoded.save('gen_e{}.midi'.format(epoch))


def start(hparams):
    ''' start
        #TODO '''
    input_vocab_size  = 128+128+100+100
    target_vocab_size = 128+128+100+100

    dataset = tf.data.Dataset.list_files('/home/big/datasets/maestro-v2.0.0/**/*.midi')

    dataset_single = pro.pipeline([
        pro.midi(),
        pro.frame(hparams['frame_size'], hparams['frame_size'], True),
        pro.unbatch(),
    ])(dataset).as_numpy_iterator()

    transformer = Transformer(input_vocab_size=input_vocab_size,
                              target_vocab_size=target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              hparams=hparams)



    seed = next(dataset_single)
    seed = np.array(seed)

    trainer = Trainer(dataset, hparams)
    ckpt = tf.train.Checkpoint(
        step=trainer.step,
        transformer=transformer,
        optimizer=transformer.optimizer
    )

    trainer.init_checkpoint(ckpt)


    generate(transformer, 4, seed)
    print('Generated sample')
