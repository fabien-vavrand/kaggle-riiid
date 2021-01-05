import os
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from riiid.core.encoders import ScoreEncoder
from riiid.core.data import save_pkl, load_pkl
from riiid.core.model import RiiidModel
from riiid.core.neural import TrainingLogs
from riiid.core.utils import update_pipeline

from riiid.saint.layers import Saint
from riiid.saint.utils import CustomSchedule, loss_function, accuracy_function, auc_function, binary_focal_loss
from riiid.saint.transformers import LecturesTransformer, QuestionsTransformer


class SaintModel:

    def __init__(self, questions=None, lectures=None, length=100):
        self.questions = questions
        self.lectures = lectures
        self.length = length
        self.pad_token = 0

        self.time_bins = 50
        self.elapsed_bins = 40  # Not used
        self.lag_bins = 40

        self.num_layers = 4
        self.d_model = 256
        self.dff = 128
        self.num_heads = 4  #8
        self.dropout = 0.3
        self.epochs = 1
        self.batch_size = 64
        self.validation_batch_size = 64
        self.warmup = 5000
        self.patience = 2
        self.use_focal_loss = False

        self.validation_ratio = 0.05

        self.model_id: str = None
        self.metadata = {}
        self.scores = {}

        self.categoricals = {
            "content_id": {"n": 13523, "min": 1, "max": 13523},
            "part": {"n": 7, "min": 1, "max": 7},
            "tag": {"n": 118, "min": 1, "max": 118},
            "tags": {"n": 1520, "min": 1, "max": 1520},
            "content_time": {"n": self.time_bins + 1, "min": 0, "max": self.time_bins},
            "question_elapsed_time": {"n": self.elapsed_bins + 1, "min": 0, "max": self.elapsed_bins},
            "question_lag_time": {"n": self.lag_bins + 1, "min": 0, "max": self.lag_bins},
            "question_had_explanation": {"n": 2, "min": 1, "max": 2},
            "tasks_since_last_lecture": {"n": 12, "min": 1, "max": 12},
            "content_id_answered_correctly": {"n": 27046, "min": 1, "max": 27046},
            "answered_correctly": {"n": 2, "min": 1, "max": 2},
        }
        self.features = {
            'content_id': np.int32, 'part': np.int32, 'tags': np.int32, 'content_time': np.int32,
            'tasks_since_last_lecture': np.int32, 'content_id_encoded': np.float32,
            'content_id_answered_correctly': np.int32, 'answered_correctly': np.int32,
            'question_lag_time': np.int32, 'question_had_explanation': np.int32, 'question_elapsed_time': np.float32
        }
        self.encoder_features = ['content_id', 'part', 'tags', 'content_time', 'tasks_since_last_lecture', 'content_id_encoded']
        self.decoder_features = ['content_id_answered_correctly', 'answered_correctly', 'question_elapsed_time', 'question_lag_time', 'question_had_explanation']

        self.prior_features = ['question_elapsed_time', 'question_lag_time', 'question_had_explanation']
        self.independent_features = ['content_id', 'part', 'tags', 'content_time', 'tasks_since_last_lecture', 'content_id_encoded']
        self.dependent_features = ['content_id_answered_correctly', 'answered_correctly']

        self.lectures_pipeline = None
        self.pipeline = None
        self.model = None
        self.context_features = None
        self.context_users = None
        self.context_priors = None

        self.previous_test = None
        self.test_batch = 0

        self._init_model()

    def get_name(self, prefix=None, ext='pkl'):
        if prefix:
            return 'saint_{}_{}'.format(self.model_id, prefix)
        else:
            return 'saint_{}.{}'.format(self.model_id, ext)

    def get_normalized_name(self):
        return 'saint_{}'.format(self.model_id).replace('_', '-')

    def fit_transform(self, X):
        logging.info('- Fit')
        self._init_fit(X)

        self.lectures_pipeline = make_pipeline(
            LecturesTransformer(self.lectures)
        )
        X = self.lectures_pipeline.fit_transform(X)
        X = RiiidModel.remove_lectures(X)

        cv = self._build_cv(X)
        self.pipeline = make_pipeline(
            ScoreEncoder('content_id', cv=cv, smoothing_min=5, smoothing_value=1, noise=0.005),
            QuestionsTransformer(self.questions, time_bins=self.time_bins, lag_bins=self.lag_bins)
        )
        X = self.pipeline.fit_transform(X)

        self._create_context(X)
        return X

    def _init_model(self):
        self.model_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    def _init_fit(self, X):
        self.metadata['n_rows'] = len(X)
        self.metadata['n_users'] = X['user_id'].nunique()

    def _build_cv(self, X):
        cv = list(GroupKFold(n_splits=5).split(X, groups=X['user_id']))
        cv = [(X.iloc[cv_train].index.values, X.iloc[cv_valid].index.values, True) for cv_train, cv_valid in cv]
        return cv

    def _create_context(self, X):
        X = X.groupby('user_id').tail(self.length)
        self.context_features = self.create_features(X)
        self.context_users = {user_id: i for i, (user_id, _) in enumerate(X.groupby('user_id'))}

        X = X.groupby(['user_id', 'task_container_id'], sort=False).size().reset_index(name='size')
        X = X.drop_duplicates('user_id', keep='last')[['user_id', 'size']]
        self.context_priors = X.set_index('user_id')['size'].to_dict()

    def split_train_test(self, X):
        user_ids = X['user_id'].unique()
        test_size = int(self.validation_ratio * len(user_ids))
        train_uids = user_ids[:-test_size]
        test_uids = user_ids[-test_size:]

        train = X[X['user_id'].isin(train_uids)]
        test = X[X['user_id'].isin(test_uids)]
        return train, test

    def create_features(self, X):
        logging.info('- Create features on {} rows'.format(len(X)))
        X_by_user = X.groupby('user_id')
        features = {}
        for feature_name in self.features:
            feature = X_by_user[feature_name].apply(lambda x: x.values)
            feature_list = []

            for values in feature.values:
                length = len(values)
                if length < self.length:
                    pad_width = self.length - length
                    values = np.pad(values, (pad_width, 0), mode='constant', constant_values=self.pad_token)
                elif length > self.length:
                    mod = length % self.length
                    if mod != 0:
                        values = np.pad(values, (self.length - mod, 0), mode='constant', constant_values=self.pad_token)
                values = values.reshape((-1, self.length))
                feature_list.append(values)
            feature_array = np.concatenate(feature_list)
            features[feature_name] = feature_array
        return features

    def create_dataset(self, features, cast_types=True):
        logging.info('- Create dataset')

        if cast_types:
            X = (
                tuple(features[f][:, 1:].astype(self.features[f]) for f in self.encoder_features),
                tuple(features[f][:, :-1].astype(self.features[f]) for f in self.decoder_features)
            )
            y = features['answered_correctly'][:, 1:].astype(np.int32)
        else:
            X = (
                tuple(features[f][:, 1:] for f in self.encoder_features),
                tuple(features[f][:, :-1] for f in self.decoder_features)
            )
            y = features['answered_correctly'][:, 1:]
        return X, y

    def cast_dataset(self, X, y):
        X = tuple(
            tuple(fvalue.astype(self.features[fname]) for fname, fvalue in zip(features, x))
            for features, x in zip([self.encoder_features, self.decoder_features], X)
        )
        y = y.astype(np.int32)
        return X, y

    def to_dataset(self, X, y, shuffle=True):
        size = len(y)
        X = tf.data.Dataset.from_tensor_slices(X)
        y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((X, y))
        if shuffle:
            dataset = dataset.shuffle(size)
        return dataset.batch(self.batch_size)

    def create_model(self):
        logging.info('- Create model')
        self.model = Saint(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            embedding_sizes={f: self._get_embedding_size(f) for f in self.categoricals},
            pe_input=self.length-1,
            pe_target=self.length-1,
            rate=self.dropout
        )
        lr = CustomSchedule(self.d_model, warmup_steps=self.warmup)
        opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
        if self.use_focal_loss:
            focal_loss_function = binary_focal_loss(gamma=2.0, alpha=0.30)
            self.model.compile(loss=focal_loss_function, optimizer=opt)
        else:
            self.model.compile(loss=loss_function, optimizer=opt)

    def _get_embedding_size(self, column):
        # The min value is usually 1, so we add 1 for the padding token
        # If the min value is 0, the padding token is already included
        return self.categoricals[column]['n'] + self.categoricals[column]['min']

    def train(self, X_train, y_train, X_test, y_test):
        self.create_model()

        logging.info('- Train')

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, mode='min'),
            TrainingLogs(metrics=['loss', 'val_loss'])
        ]

        train_ds = self.to_dataset(X_train, y_train, shuffle=True)
        test_ds = self.to_dataset(X_test, y_test, shuffle=False)
        history = self.model.fit(train_ds, validation_data=test_ds, epochs=self.epochs, callbacks=my_callbacks, verbose=0)

    def score(self, X, y, keep_last=None):
        y_pred = self.model.predict(X, batch_size=self.validation_batch_size, verbose=0)
        auc = auc_function(y, y_pred, keep_last=keep_last)
        logging.info('AUC (last {}) = {:.2%}'.format(keep_last, auc))
        return auc

    def update(self, test):
        prior_user_answer = eval(test['prior_group_responses'].values[0])
        prior_answered_correctly = eval(test['prior_group_answers_correct'].values[0])
        test = test.drop(columns=['prior_group_answers_correct', 'prior_group_responses'])

        if self.previous_test is not None:
            self.previous_test['user_answer'] = prior_user_answer
            self.previous_test['answered_correctly'] = prior_answered_correctly

            X = self.previous_test
            # X = update_pipeline(self.lectures_pipeline, X)  # Not required
            X = RiiidModel.remove_lectures(X)
            if len(X) > 0:
                y = X['answered_correctly']
                X = update_pipeline(self.pipeline, X, y)
                self._update_context(X, self.dependent_features)

        self.previous_test = test.copy()
        return test

    def _update_context(self, X, features):
        user_id = X['user_id'].values
        values = {f: X[f].values for f in features}

        for r in range(len(X)):
            try:
                idx = self.context_users[user_id[r]]
                for f in features:
                    self.context_features[f][idx, :] = np.roll(self.context_features[f][idx, :], -1)
                    self.context_features[f][idx, -1] = values[f][r]
            except KeyError:
                # new user with index = current length, or new length - 1
                idx = self.context_features['content_id'].shape[0]
                self.context_users[user_id[r]] = idx
                for f in self.features:
                    ftype = self.context_features[f].dtype
                    self.context_features[f] = np.vstack((self.context_features[f], np.zeros((1, self.length), dtype=ftype)))
                for f in features:
                    self.context_features[f][idx, -1] = values[f][r]

    def _update_context_with_priors(self, X):
        user_id = X['user_id'].values
        values = {f: X[f].values for f in self.prior_features}
        context = {}

        for r in range(len(X)):
            if user_id[r] not in context:
                context[user_id[r]] = 0

                # We only update priors once per user
                try:
                    idx = self.context_users[user_id[r]]
                    task_size = self.context_priors[user_id[r]]
                    for f in self.prior_features:
                        #self.context_features[f][idx, :] = np.roll(self.context_features[f][idx, :], -task_size)
                        for i in range(task_size):
                            self.context_features[f][idx, -i - 1] = values[f][r]
                except KeyError:
                    # Either we know the user, or there are no priors
                    pass

            context[user_id[r]] += 1

        # Update context
        self.context_priors.update(context)

    def _roll_context_on_priors(self, X):
        for user_id in X['user_id'].values:
            task_size = self.context_priors[user_id]
            try:
                idx = self.context_users[user_id]
                for f in self.prior_features:
                    # Warning: last values are wrong because of roll, but will be overwritten before being used
                    self.context_features[f][idx, :] = np.roll(self.context_features[f][idx, :], -task_size)
            except KeyError:
                # New user, priors are empty
                pass

    def predict(self, X):
        if self.test_batch % 1 == 0:
            logging.info('Running test batch {}'.format(self.test_batch))

        X = self.lectures_pipeline.transform(X)
        X = RiiidModel.remove_lectures(X)
        if len(X) > 0:
            predictions = X[['row_id']].copy()
            X = self.pipeline.transform(X)
            self._update_context_with_priors(X)
            inputs = self._create_prediction_data(X)
            self._roll_context_on_priors(X)
            self._update_context(X, self.independent_features)
            predictions['answered_correctly'] = self.model.predict(inputs)[:, -1, -1]
        else:
            predictions = pd.DataFrame(columns=['row_id', 'answered_correctly'])
        self.test_batch += 1
        return X, predictions

    def _create_prediction_data(self, X):
        # Warning: this only works if padding token is 0
        # else, we should use np.full((,), self.padding_token)

        user_id = X['user_id'].values
        encoder_inputs = tuple(np.zeros((len(X), self.length-1), dtype=self.features[f]) for f in self.encoder_features)
        decoder_inputs = tuple(np.zeros((len(X), self.length-1), dtype=self.features[f]) for f in self.decoder_features)
        values = {f: X[f].values for f in self.encoder_features}
        for r in range(len(X)):
            try:
                idx = self.context_users[user_id[r]]
                for i, f in enumerate(self.encoder_features):
                    encoder_inputs[i][r, :-1] = self.context_features[f][idx, 2:]
                for i, f in enumerate(self.decoder_features):
                    decoder_inputs[i][r, :] = self.context_features[f][idx, 1:]
            except KeyError:
                # inputs already filled with padding token
                pass

            # we fill last values
            for i, f in enumerate(self.encoder_features):
                encoder_inputs[i][r, -1] = values[f][r]

        return (encoder_inputs, decoder_inputs)

    def save(self, path):
        if self.model is not None:
            self.model.save(os.path.join(path, self.get_name('model')))
            self.model = None
        save_pkl(self, os.path.join(path, self.get_name()))

    def load_model_from_path(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={
            'CustomSchedule': CustomSchedule,
            'loss_function': loss_function,
            'focal_loss_function': binary_focal_loss(gamma=2.0, alpha=0.30)
        })

    @staticmethod
    def load(path, model_id):
        model = load_pkl(os.path.join(path, f'{model_id}.pkl'))
        model_path = os.path.join(path, f'{model_id}_model')
        if os.path.isdir(model_path):
            model.load_model_from_path(model_path)
        return model
