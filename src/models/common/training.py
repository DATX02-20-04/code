import tensorflow as tf
import os
import time
import datetime

class Trainer():
    def __init__(self, dataset, hparams):
        self.dataset = dataset
        self.hparams = hparams
        self.step = tf.Variable(0)
        self.train_summary_writer = None

    def init_tensorboard(self):
        # Tensorfboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f"./logs/{self.hparams['name']}/{current_time}/train/"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def init_checkpoint(self, ckpt):
        self.ckpt = ckpt
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  os.path.join(self.hparams['save_dir'], 'ckpts', self.hparams['name']),
                                                  max_to_keep=3)

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def set_train_step(self, train_step):
        self.train_step = train_step

    def on_epoch_start(self, epoch, step, tsw=None):
        pass

    def on_step(self, epoch, step, stats, tsw=None):
        pass

    def on_epoch_complete(self, epoch, step, duration, tsw=None):
        print(f'Epoch {epoch} completed in {duration:.3f} seconds at step {step}')

    def run(self):
        if self.train_step is None:
            raise Exception("No train_step specified, call set_train_step on the trainer with your training step.")

        steps_per_epoch = self.hparams['steps'] if 'steps' in self.hparams else None

        for epoch in range(1, self.hparams['epochs']+1):
            start = time.time()
            self.on_epoch_start(epoch, self.step.numpy(), tsw=self.train_summary_writer)

            d = self.dataset.take(steps_per_epoch) if steps_per_epoch > 0 else self.dataset

            for batch in d:
                self.step.assign_add(1)
                stats = self.train_step(batch)
                self.on_step(epoch, self.step.numpy(), stats, tsw=self.train_summary_writer)

            if self.ckpt is not None:
                self.manager.save()

            end = time.time()
            duration = end - start
            self.on_epoch_complete(epoch, self.step.numpy(), duration, tsw=self.train_summary_writer)
