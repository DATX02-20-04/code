import tensorflow as tf
import numpy as np
import data.process as pro
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class SineNet():
    def __init__(self, hparams, ft_shape):
        self.hparams = hparams
        self.ft_shape = ft_shape

        self.param_net = self.create_param_net()
        self.optimizer = tfk.optimizers.Adam(hparams['lr'])

    @tf.function
    def get_loss(self, y_target, params):
        wave = self.get_wave(params)
        #loss = tfk.losses.mean_squared_error(y_target, wave)

        specloss = tfk.losses.mean_squared_error(y_target, wave)

        return specloss

    @tf.function
    def train_step(self, x):
        x, y_target = x

        with tf.GradientTape() as tape:
            params = self.param_net(x, training=True)
            
            loss = self.get_loss(y_target, params)


        gradients = tape.gradient(loss, self.param_net.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.param_net.trainable_variables))

        return loss

    def create_param_net(self):
        i = tfkl.Input(shape=self.ft_shape)

        #o = tfkl.Reshape([*self.ft_shape, 1])(i)
        #o = tfkl.Conv2D(8, 3, activation='relu')(o)
        #o = tfkl.Conv2D(8, 3, strides=2, activation='relu')(o)
        #o = tfkl.Conv2D(8, 3, strides=2, activation='relu')(o)
        #o = tfkl.Flatten()(o)
        o = tfkl.Dense(256, activation='relu')(i)
        o = tfkl.Dense(self.hparams['channels']*self.hparams['params'], activation='sigmoid')(o)

        return tfk.Model(inputs=i, outputs=o)

