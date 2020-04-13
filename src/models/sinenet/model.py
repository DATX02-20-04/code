import tensorflow as tf
import numpy as np
import data.process as pro
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class SineNet():
    def __init__(self, hparams):
        self.hparams = hparams

        self.model = self.create_model()
        self.optimizer = tfk.optimizers.Adam(hparams['lr'])

    @tf.function
    def train_step(self, x):
        pitch, hist, next_sample = x

        with tf.GradientTape() as tape:
            gen_sample = self.model([pitch, hist], training=True)
            
            loss = tfk.losses.mean_squared_error(next_sample, gen_sample)


        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def create_model(self):
        pitch_i = tfkl.Input(shape=(self.hparams['pitches'],))
        hist_i = tfkl.Input(shape=(self.hparams['history'],))

        pitch = tfkl.Dense(128, activation='relu', use_bias=False)(pitch_i)

        hist = tfkl.Reshape([self.hparams['history'], 1])(hist_i)
        hist = tfkl.Conv1D(16, 4, 2, activation='relu')(hist)
        hist = tfkl.Conv1D(64, 4, 2, activation='relu')(hist)
        hist = tfkl.Flatten()(hist)
        hist = tfkl.Dense(128, activation='relu', use_bias=False)(hist_i)

        o = tfkl.Concatenate(1)([pitch, hist])

        o = tfkl.Dense(128, activation='relu', use_bias=False)(o)
        o = tfkl.Dense(64, activation='relu', use_bias=False)(o)
        o = tfkl.Dense(1, activation='tanh')(o)

        return tfk.Model(inputs=[pitch_i, hist_i], outputs=o)
