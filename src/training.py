import tensorflow as tf

def create_train_loop(dataset, train_step, epochs, steps, ckpt, save_dir, on_epoch_start=None, on_step=None, on_epoch_complete=None):
    manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    def train():
        step = 0
        for epoch in range(1, epochs+1):
            if on_epoch_start is not None:
                on_epoch_start(epoch, step)

            for batch in dataset.take(steps):
                step += 1
                stats = train_step(batch)
                if on_step is not None:
                    on_step(step, stats)

            self.manager.save()
            if on_epoch_complete is not None:
                on_epoch_complete(epoch, step)

    return train

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
