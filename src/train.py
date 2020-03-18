import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import argparse
import subprocess
import os
import importlib
from util import load_hparams, parse_train_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start training a model')

    parser.add_argument('--model', metavar='N', required=True, help='The model to train', type=str, nargs='?')

    args, unknownargs = parser.parse_known_args()

    train = importlib.import_module(f'models.{args.model}.train')

    # Load hyperparams from yaml file
    hparams = load_hparams(f'hparams/{args.model}.yml')
    hparams = parse_train_args(unknownargs, hparams)

    train.start(hparams)
