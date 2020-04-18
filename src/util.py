import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
import argparse
import io

def load_hparams(path):
    with open(path, 'r') as stream:
        try:
            hparams = yaml.safe_load(stream)['hparams']
        except yaml.YAMLError as e:
            print(e)

    return hparams


def parse_train_args(args, hparams):
    parser = argparse.ArgumentParser(description='Start training of the model')
    parser.add_argument('--save_dir', metavar='PATH', help='Set the save directory for images and checkpoints', type=str, nargs='?', default=hparams['save_dir'])
    parser.add_argument('--epochs', metavar='N', help='Set the number of epochs to train', type=int, nargs='?', default=hparams['epochs'])
    parser.add_argument('--steps', metavar='N', help='Set the number of steps per epoch', type=int, nargs='?', default=hparams['steps'])
    parser.add_argument('--batch_size', metavar='N', help='Set the batch size', type=int, nargs='?', default=hparams['batch_size'])
    parser.add_argument('--buffer_size', metavar='N', help='Set the buffer size', type=int, nargs='?', default=hparams['buffer_size'])

    args = vars(parser.parse_args(args))

    return { **hparams, **args }

def get_plot_image():
    buf = io.BytesIO()
    plt.savefig(buf,  format='png')
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
