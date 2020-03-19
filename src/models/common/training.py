import tensorflow as tf
import os
import time
import progressbar

class Trainer():
    def __init__(self, dataset, hparams):
        self.dataset = dataset
        self.hparams = hparams
        self.step = tf.Variable(0)

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

    def on_epoch_start(self, epoch, step):
        pass

    def on_step(self, epoch, step, stats):
        pass

    def on_epoch_complete(self, epoch, step, duration):
        print(f'Epoch {epoch} completed in {duration:.3f} seconds at step {step}')

    def run(self):

        epochs_bar = progressbar.ProgressBar(
            name="Epochs",
            min_value=1,
            max_value=self.hparams['epochs']+1,
            redirect_stdout=True
        )

        if self.train_step is None:
            raise Exception("No train_step specified, call set_train_step on the trainer with your training step.")

        steps_per_epoch = self.hparams['steps'] if 'steps' in self.hparams else None

        for epoch in range(1, self.hparams['epochs']+1):
            start = time.time()
            epochs_bar.update(epoch)
            self.on_epoch_start(epoch, self.step.numpy())

            d = self.dataset.take(steps_per_epoch) if steps_per_epoch > 0 else self.dataset

            for batch in d:
                self.step.assign_add(1)
                stats = self.train_step(batch)
                self.on_step(epoch, self.step.numpy(), stats)

            if self.ckpt is not None:
                self.manager.save()

            end = time.time()
            duration = end - start
            self.on_epoch_complete(epoch, self.step.numpy(), duration)
