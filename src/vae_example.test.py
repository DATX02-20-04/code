#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
import tensorflow as tf
from vae_example import VAEExample

if __name__ == '__main__':
    hparams = {
        'epochs': 2,
        'sample_rate': 16000,
        'window_samples': 128,
        'steps_per_epoch': None,
        'batch_size': 8,
        'buffer_size': 100,
        'latent_size': 2,
        'model_scale': 8,
        'lr': 1e-3,
        'log_amin': 1e-5,
        'num_examples': 1,
        'save_dir': '.'
    }

    dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([128, 1000]))

    vae = VAEExample(dataset, hparams)
    vae.train()

    print("--- TEST COMPLETE ---")
