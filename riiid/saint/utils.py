import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.cast(tf.math.rsqrt(step), tf.float32)
        arg2 = tf.cast(step * (self.warmup_steps ** -1.5), tf.float32)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model.numpy(),
            'warmup_steps':self.warmup_steps
        }
        return config


def bce(real, pred):
    real = tf.cast(real, tf.float32)
    p1 = real * tf.math.log(pred)
    p2 = (1 - real) * tf.math.log(1 - pred)
    return -1.0 * (p1 + p2)


def loss_function(real, pred):
    pred = pred[:, :, -1]
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    real = real - 1
    loss_ = bce(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(tf.cast(real, 'int64'), tf.argmax(pred, axis=-1))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def auc_function(real, pred, keep_last=None):
    if keep_last:
        real = real[:, -keep_last:]
        pred = pred[:, -keep_last:, :]
    real = real.ravel()
    pred = pred[:, :, -1].ravel()
    mask = real != 0
    real = real[mask] - 1
    pred = pred[mask]
    return roc_auc_score(real, pred)


def cast_to_int32(data):
    if isinstance(data, np.ndarray):
        if pd.api.types.is_integer_dtype(data) and data.dtype != np.int32:
            return data.astype(np.int32)
        elif pd.api.types.is_float_dtype(data) and data.dtype != np.float32:
            return data.astype(np.float32)
        else:
            return data
    else:
        return tuple([cast_to_int32(d) for d in data])