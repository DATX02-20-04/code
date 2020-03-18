#!/usr/bin/env python3
from training import Trainer

class Model():
    def __init__(self, dataset, hparams):
        self.hparams = hparams
        self.dataset = self.preprocess(dataset)

        self.trainer = Trainer(self.dataset, self.hparams)
        self.trainer.on_epoch_start = self.on_epoch_start
        self.trainer.on_step = self.on_step
        self.trainer.on_epoch_complete = self.on_epoch_complete

    def train(self):
        self.trainer.run()

    def preprocess(self, dataset):
        return dataset

    def on_epoch_start(self, epoch, step):
        pass

    def on_step(self, step, stats):
        pass

    def on_epoch_complete(self, epoch, step, duration):
        pass
