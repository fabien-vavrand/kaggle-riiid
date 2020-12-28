import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from riiid.core.data import save_pkl, load_pkl
from riiid.saint.layers import Saint
from riiid.saint.utils import CustomSchedule, accuracy, loss_function



class SaintModel:

    def __init__(self, questions, lectures, length=100):
        self.questions = questions
        self.lectures = lectures
        self.length = length
        self.pad_token = 0

        self.num_layers = 4
        self.d_model = 156
        self.dff = 64
        self.num_heads = 4  #8
        self.dropout = 0.1
        self.epochs = 40
        self.batch_size = 128
        self.warmup = 5000
        self.use_tpu = False

        self.model_id: str = None
        self.metadata = {}

        self.model = None
        self.context = None

    def get_name(self, prefix=None):
        if prefix:
            return 'saint_{}_{}'.format(self.model_id, prefix)
        else:
            return 'saint_{}.zip'.format(self.model_id)

    def fit(self, X):
        logging.info('- Fit')
        self._init_fit(X)

        X = X[X['content_type_id'] == 0].copy()
        X.replace([np.nan], 0, inplace=True)

        # Rebase ids to reserve 0 for padding token
        # part is already > 0
        self.questions['question_id'] += 1
        X['content_id'] += 1
        X['answered_correctly'] += 1

        X = pd.merge(X, self.questions, left_on='content_id', right_on='question_id', how='left')

        user_ids = X['user_id'].unique()
        test_size = int(0.1 * len(user_ids))
        train_uids = user_ids[:-test_size]
        test_uids = user_ids[-test_size:]

        train_set = X[X['user_id'].isin(train_uids)]
        test_set = X[X['user_id'].isin(test_uids)]

        train_features = self.create_features(train_set)
        test_features = self.create_features(test_set)

        train_ds = self.create_dataset(train_features)
        test_ds = self.create_dataset(test_features)

        self.context = self.create_user_data(X)
        return train_ds, test_ds

    def _init_fit(self, X):
        self.model_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metadata['n_rows'] = len(X)
        self.metadata['n_users'] = X['user_id'].nunique()

    def create_features(self, df):
        logging.info('- Create features on {} rows'.format(len(df)))
        features_names = ['content_id', 'part', 'answered_correctly']
        df_by_user = df.groupby('user_id')
        features = {}
        for feature_name in features_names:
            feature = df_by_user[feature_name].apply(lambda x: x.values)
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

    def create_user_data(self, df):
        # df = df[df.content_type_id == 0]
        df = df[['content_id', 'user_id', 'answered_correctly', 'part']]
        udf = df.groupby('user_id').tail(self.length)
        udf = udf.groupby('user_id')
        vals = udf.apply(lambda x: x.values)
        udata = {}
        for uid, v in vals.iteritems():
            udata[uid] = [v[:, 0], v[:, 2], v[:, 3]]
            ud = udata[uid]
            nin = ud[0].shape[0]
            if nin > self.length:
                for i in range(3):
                    ud[i] = ud[i][-self.length:]
            elif nin < self.length:
                diff = self.length - nin
                for i in range(3):
                    ud[i] = np.pad(ud[i], (diff, 0), mode='constant', constant_values=self.pad_token)
            udata[uid] = ud

        for k, v in udata.items():
            udata[k] = [x + 1 for x in v]
        return udata

    def create_dataset(self, features):
        logging.info('- Create dataset')
        X = (
            (
                 features['content_id'][:, 1:].astype('int32'),
                 features['part'][:, 1:].astype('int32')
            ),
            features['answered_correctly'][:, :-1].astype('int32')
        )
        y = features['answered_correctly'][:, 1:].astype('int32')
        size = len(y)

        X = tf.data.Dataset.from_tensor_slices(X)
        y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((X, y)).shuffle(size).batch(self.batch_size)
        return dataset

    def create_model(self):
        self.model = Saint(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            n_contents=len(self.questions['question_id'].unique()) + 1,  # +1 for padding token
            n_parts=len(self.questions['part'].unique()) + 1,  # +1 for padding token
            n_answers=3,
            pe_input=self.length-1,
            pe_target=self.length-1,
            rate=self.dropout
        )
        lr = CustomSchedule(self.d_model, warmup_steps=self.warmup)
        opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
        self.model.compile(loss=loss_function, optimizer=opt)  # metrics=[accuracy]

    def train(self, train_ds, test_ds):
        logging.info('- Train')
        if self.use_tpu:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
            with tpu_strategy.scope():
                self.create_model()
        else:
            self.create_model()

        self.history = self.model.fit(train_ds, validation_data=test_ds, epochs=self.epochs)

    def score(self, y_true, test_ds):
        val_preds = self.model.predict(test_ds, verbose=1)
        y_pred = val_preds[:, -1, -1]
        # y_true = Y_dev[:, -1] - 1
        # val_auc = roc_auc_score(y_true, y_pred)

    def predict(self, X):
        raise NotImplementedError()

    def save(self, path=None):
        #self.model.save('gs://riiid_models/{}'.format(self.get_name()))
        self.model = None
        self.history = None
        save_pkl(self, path)

    @staticmethod
    def load(path):
        model = load_pkl(path)
        model.model = tf.keras.models.load_model(path, custom_objects={
            'accuracy': accuracy,
            'CustomSchedule': CustomSchedule,
            'loss_function': loss_function
        })
        return model
