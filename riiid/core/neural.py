import os
import io
import scipy
import pickle
import zipfile
import logging
import tempfile
import numpy as np

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = 3

class ScalingTransformer:

    def __init__(self, min_unique_values=5, skewness_threshold=1, max_rows=5_000_000):
        self.min_unique_values = min_unique_values
        self.skewness_threshold = skewness_threshold
        self.max_rows = max_rows
        self.rows = None
        self.columns = None
        self.standard_features = None
        self.skewed_features = None
        self.standard_scaler = None
        self.power_scaler = None

    def fit(self, X, y=None):
        logging.info('- Fit scaling transformer')
        self.rows, self.columns = X.shape
        self.standard_features = []
        self.skewed_features = []
        self.standard_scaler = StandardScaler()
        self.power_scaler = PowerTransformer()

        for i in range(self.columns):
            n_uniques = len(np.unique(X[:, i]))
            if n_uniques <= self.min_unique_values:
                self.standard_features.append(i)
            else:
                skewness = scipy.stats.skew(X[:, i])
                if skewness > self.skewness_threshold:
                    self.skewed_features.append(i)
                else:
                    self.standard_features.append(i)

        self.standard_features = np.array(self.standard_features)
        self.skewed_features = np.array(self.skewed_features)
        logging.info('{} standard features'.format(len(self.standard_features)))
        logging.info('{} skewed features'.format(len(self.skewed_features)))

        if self.rows > self.max_rows:
            X = X.sample(n=self.max_rows)

        self.standard_scaler.fit(X[:, self.standard_features])
        self.power_scaler.fit(X[:, self.skewed_features])
        return self

    def transform(self, X):
        return np.hstack([
            self.standard_scaler.transform(X[:, self.standard_features]),
            self.power_scaler.transform(X[:, self.skewed_features]),
        ])


class TrainingLogs(keras.callbacks.Callback):

    def __init__(self, metrics):
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs=None):
        text = ' - '.join(['{}: {:.4f}'.format(metric, logs[metric]) for metric in self.metrics])
        logging.info(f'[epoch {epoch}] {text}')


class NeuralModel:

    def __init__(self, params):
        self.params = params
        self.pipeline = None
        self.input_size = None
        self.model = None
        self.scores = {}

        logging.info('- Available devices:')
        for device in tf.config.list_physical_devices():
            logging.info(device)

    def fit(self, X_train, y_train, X_valid, y_valid):
        logging.info('- Fitting mlp pipeline')
        self.pipeline = make_pipeline(
            SimpleImputer(strategy='median', add_indicator=True),
            ScalingTransformer()
        )

        X_train = self.pipeline.fit_transform(X_train)
        X_valid = self.pipeline.transform(X_valid)

        logging.info('- Fitting mlp model')
        self.input_size = X_train.shape[1]
        self.init_model()
        K.clear_session()
        my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True, mode='max'),
            TrainingLogs(metrics=['loss', 'auc', 'val_loss', 'val_auc'])
        ]
        self.model.fit(
            X_train, y_train,
            epochs=self.params['epochs'], batch_size=self.params['batch_size'],
            validation_data=(X_valid, y_valid),
            callbacks=my_callbacks,
            verbose=0
        )
        train_loss, train_auc = self.model.evaluate(X_train, y_train, batch_size=self.params['batch_size'], verbose=0)
        val_loss, val_auc = self.model.evaluate(X_valid, y_valid, batch_size=self.params['batch_size'], verbose=0)

        self.scores['train_loss'] = train_loss
        self.scores['train_auc'] = train_auc
        self.scores['val_loss'] = val_loss
        self.scores['val_auc'] = val_auc

    def init_model(self):
        model = keras.Sequential([keras.Input(shape=(self.input_size,))])
        for size in self.params['layers']:
            model.add(layers.Dense(size, activation=tf.nn.relu))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.params['dropout']))

        model.add(layers.Dense(1, activation=tf.nn.sigmoid))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
        )

        self.model = model

    def predict(self, X):
        X = self.pipeline.transform(X)
        y = self.model.predict(X)[:,0]
        return y

    def save(self, path=None):
        with tempfile.NamedTemporaryFile() as file:
            temp_file = file.name

        self.model.save(temp_file, save_format='h5')
        self.model = None

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.write(temp_file, 'model.h5')
            zip.writestr('neural.pkl', pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

        os.remove(temp_file)

        if path is None:
            return zip_buffer
        else:
            with open(path, 'wb') as file:
                file.write(zip_buffer.getvalue())

    @staticmethod
    def load(path):
        temp_dir = tempfile.mkdtemp()
        temp_h5_file = os.path.join(temp_dir, 'model.h5')

        with zipfile.ZipFile(path, 'r') as zip:
            model = pickle.loads(zip.read('neural.pkl'))
            zip.extract('model.h5', temp_dir)

        model.model = keras.models.load_model(temp_h5_file)

        os.remove(temp_h5_file)
        os.removedirs(temp_dir)
        return model
