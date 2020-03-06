import tensorflow as tf
import os
import time

class Trainer():
    def __init__(self, dataset, hparams):
        self.dataset = dataset
        self.hparams = hparams
        self.step = tf.Variable(0)

    def init_checkpoint(self, ckpt):
        self.ckpt = ckpt
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(self.hparams['save_dir'], 'ckpts'), max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def set_train_step(self, train_step):
        self.train_step = train_step

    def run(self):
        steps_per_epoch = self.hparams['steps_per_epoch'] if 'steps_per_epoch' in self.hparams else None

        for epoch in range(1, self.hparams['epochs']+1):
            start = time.time()
            if self.on_epoch_start is not None:
                self.on_epoch_start(epoch, self.step)

            d = dataset.take(steps_per_epoch) if steps_per_epoch is not None else dataset

            for batch in d:
                self.step.assign_add(1)
                stats = train_step(batch)
                if self.on_step is not None:
                    self.on_step(self.step, stats)

            if self.ckpt is not None:
                self.manager.save()

            end = time.time()
            duration = end - start
            if self.on_epoch_complete is not None:
                self.on_epoch_complete(epoch, self.step, duration)

def create_gan_train_step(generator,
                          discriminator,
                          generator_loss,
                          discriminator_loss,
                          generator_optimizer,
                          discriminator_optimizer,
                          batch_size,
                          latent_size):
    @tf.function
    def train_step(x):
        noise = tf.random.normal([batch_size, latent_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_x = generator(noise, training=True)

            real_output = discriminator(x, training=True)
            fake_output = discriminator(generated_x, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    return train_step
