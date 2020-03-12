#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
import tensorflow as tf
from gan_example import GANExample

if __name__ == '__main__':
    # Setup hyperparameters
    hparams = {
        'epochs': 2,
        'steps_per_epoch': 1000,
        'sample_rate': 16000,
        'batch_size': 4,
        'buffer_size': 1000,
        'latent_size': 100,
        'generator_scale': 16,
        'discriminator_scale': 16,
        'gen_lr': 0.0001,
        'disc_lr': 0.0004,
        'log_amin': 1e-5,
        'num_examples': 16,
        'save_dir': '.',
        'plot': False
    }

    dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([16, 128, 128]))

    gan = GANExample(dataset, hparams)
    gan.train()

    print("--- TEST COMPLETE ---")
