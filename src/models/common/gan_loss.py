import tensorflow as tf

def create_simple_gan_loss(cross_entropy):
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(
            tf.zeros_like(real_output)+tf.random.uniform(real_output.shape, 0, 0.1), # Add random to smooth real labels
            real_output)
        fake_loss = cross_entropy(
            tf.ones_like(fake_output)-tf.random.uniform(fake_output.shape, 0, 0.1), # Subtract random to smooth fake labels
            fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.zeros_like(fake_output), fake_output)

    return generator_loss, discriminator_loss
