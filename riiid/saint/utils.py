import tensorflow as tf


# Noam Schedule
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
        config = {"d_model": self.d_model.numpy(), "warmup_steps": self.warmup_steps}
        return config


def bce(real, pred):
    real = tf.cast(real, tf.float32)
    p1 = real * tf.math.log(pred)
    p2 = (1 - real) * tf.math.log(1 - pred)
    return -1.0 * (p1 + p2)


def loss_function(real, pred):

    #     real = real[:,-1]
    #     pred = pred[:,-1,:]
    pred = pred[:, :, -1]
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    real = real - 1
    #     real = tf.expand_dims(real,axis=-1)
    #     pred = tf.expand_dims(pred,axis=-1)
    #     loss_ = loss_object(real, pred)
    loss_ = bce(real, pred)
    # print(loss_)
    mask = tf.cast(mask, dtype=loss_.dtype)
    #     print(loss_.shape)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy(real, pred):
    #     pred = tf.cast(pred,'int32')
    #     real = real[:,-1]
    #     pred = pred[:,-1,:]
    pred = pred[:, :, -1]
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    real = real - 1
    pred = pred > 0.5
    pred = tf.cast(pred, tf.int32)
    accuracies = tf.equal(tf.cast(real, "int32"), pred)
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
