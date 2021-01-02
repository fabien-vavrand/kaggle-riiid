import tensorflow as tf
from riiid.core.data import load_pkl
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging


configure_console_logging()

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    model = SaintModel()
    model.batch_size = 1024  # 16 * tpu_strategy.num_replicas_in_sync
    model.epochs = 50
    X_train, y_train, X_test, y_test = load_pkl('/kaggle/input/riiid-saint-features-v0/data.pkl')

    model.train(X_train, y_train, X_test, y_test)
    model.model.save('gs://riiid-models/{}'.format(model.get_name('model')))
    model.score(X_test, y_test)
