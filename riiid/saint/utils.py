import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
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


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_pred = y_pred[:, :, -1]
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        y_true = y_true - 1
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)
        loss_ = -(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - (
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    return binary_focal_loss_fixed


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