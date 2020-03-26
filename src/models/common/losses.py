import tensorflow as tf
from keras import backend

def wasserstein_loss(y_true, y_pred)
    return backend.mean(y_true * y_pred)
