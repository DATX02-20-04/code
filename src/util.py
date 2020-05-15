import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
import argparse
import io
import time

def load_hparams(path):
    with open(path, 'r') as stream:
        try:
            hparams = yaml.safe_load(stream)['hparams']
        except yaml.YAMLError as e:
            print(e)

    return hparams


def parse_train_args(args, hparams):
    parser = argparse.ArgumentParser(description='Start training of the model')
    save_dir = hparams['save_dir'] if 'save_dir' in hparams else '.'
    parser.add_argument('--save_dir', metavar='PATH', help='Set the save directory for images and checkpoints', type=str, nargs='?', default=save_dir)
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

def create_span(logger):
    spans = {}
    get_t = lambda: int(time.time*1000)
    def span(cmd, name):
        if cmd == 'start':
            if name not in spans:
                t = get_t()
                logger(f"Starting {name} at {t:.2f}...")
                spans[name] = {
                    'start': t
                }
                return t
        elif cmd == 'end':
            if name in spans:
                t = get_t()
                spans[name]['end'] = t
                elapsed = spans[name]['end'] - spans[name]['start']
                logger(f"Completed {name} in {elapsed:.3f}ms...")
                return spans[name]

    return span

def create_logger(module):
    def logger(msg, level='info'):
        print(f"[{level.upper()}] {int(time.time()*1000)} {module}: {msg}")
    return logger
