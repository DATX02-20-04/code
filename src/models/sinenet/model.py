import tensorflow as tf
import numpy as np
import data.process as pro
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class SineNet(tfk.Model):
    def __init__(self, hparams):
        super(SineNet, self).__init__()
        self.hparams = hparams

        self.models = [self.create_model() for _ in range(hparams['pitches'])]
        self.optimizer = tfk.optimizers.Adam(hparams['lr'])

    @tf.function
    def train_step(self, x):
        hists, next_samples = x

        total_loss = 0
        for i in range(self.hparams['pitches']):
            model = self.models[i]
            hist = hists[:, i, :]
            next_sample = next_samples[:, i]
            with tf.GradientTape() as tape:
                gen_sample = model(hist, training=True)

                loss = tfk.losses.mean_squared_error(next_sample, gen_sample)

            gradients = tape.gradient(loss, model.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss

        return total_loss

    def create_model(self):
        # pitch_i = tfkl.Input(shape=(self.hparams['pitches'],))
        hist_i = tfkl.Input(shape=(self.hparams['history'],))

        # pitch = tfkl.Dense(16, activation='relu')(pitch_i)

        hist = tfkl.Reshape([self.hparams['history'], 1])(hist_i)
        hist = tfkl.LSTM(10, activation='tanh')(hist)
        # hist = tfkl.Conv1D(64, 4, 2, activation='relu')(hist)
        # hist = tfkl.Flatten()(hist)
        # hist = tfkl.Dense(16, activation='relu')(hist_i)

        # o = tfkl.Concatenate(1)([pitch, hist])
        o = hist

        # o = tfkl.Dense(8, activation='tanh')(o)
        o = tfkl.Dense(1, activation='tanh')(o)

        return tfk.Model(inputs=hist_i, outputs=o)
