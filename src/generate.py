import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
import argparse
import subprocess
import os
import importlib
from util import load_hparams, parse_train_args

# Some compatability options for some graphics cards
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from a model')

    parser.add_argument('--model', metavar='N', required=True, help='The model to generate with', type=str, nargs='?')

    args, unknownargs = parser.parse_known_args()

    generate = importlib.import_module(f'models.{args.model}.generate')

    # Load hyperparams from yaml file
    hparams = load_hparams(f'hparams/{args.model}.yml')
    hparams = parse_train_args(unknownargs, hparams)

    generate.start(hparams)
