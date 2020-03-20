import tensorflow as tf
import os
import time
import progressbar as pb
import progressbar.widgets as pbw

class VariableBar(pbw.Bar, pbw.VariableMixin):
    """
    A Bar reading from a different variable. Due to limitations of ProgressBar2,
    the maximum value cannot be a second variable.
    """

    def __init__(self, variable, max_value=pb.UnknownLength, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pbw.VariableMixin.__init__(self, variable)
        self.max_value = max_value

    def __call__(self, progress, data, width):
        (prev_value, prev_max) = (progress.value, progress.max_value)
        try:
            progress.value = data['variables'][self.name] or 0 # That is a bad API
            progress.max_value = self.max_value
            return super().__call__(progress, data, width)
        finally:
            (progress.value, progress.max_value) = (prev_value, prev_max)

class Percentage(pbw.Percentage):
    """
    The default Percentage class shows N/A at zero percent, this one just shows 0%.
    """

    def __call__(self, progress, data, format=None):
        return pbw.FormatWidgetMixin.__call__(self, progress, data)

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
        if self.train_step is None:
            raise Exception("No train_step specified, call set_train_step on the trainer with your training step.")

        with pb.ProgressBar(
                widgets=[
                    Percentage(),
                    ' ', pbw.SimpleProgress(format='(%s)' % pb.SimpleProgress.DEFAULT_FORMAT),
                    ' ', pbw.Bar(),
                    ' ', VariableBar("batch", self.hparams.get('steps', pb.UnknownLength)),
                    ' ', pbw.Timer(),
                    ' ', pbw.AdaptiveETA(),
                ],
                min_value=1,
                max_value=self.hparams['epochs']+1,
                redirect_stdout=True,
        ).start() as bar:
            for epoch in range(1, self.hparams['epochs']+1):
                start = time.time()
                self.on_epoch_start(epoch, self.step.numpy())
                bar.update(epoch)

                d = self.dataset.take(self.hparams['steps']) if 'steps' in self.hparams else self.dataset

                for batch_no, batch in enumerate(d):
                    bar.update(batch=batch_no)
                    self.step.assign_add(1)
                    stats = self.train_step(batch)
                    self.on_step(epoch, self.step.numpy(), stats)

                if self.ckpt is not None:
                    self.manager.save()

                end = time.time()
                duration = end - start
                self.on_epoch_complete(epoch, self.step.numpy(), duration)
