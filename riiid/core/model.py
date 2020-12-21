import io
import os
import json
import pickle
import zipfile
import datetime
import logging
import typing as t
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline

from riiid.utils import downcast_int, logging_callback, keys_to_int, get_riiid_directory
from riiid.config import FLOAT_DTYPE
from riiid.core.data import load_pkl
from riiid.core.utils import update_pipeline, DataFrameAnalyzer
from riiid.core.encoders import ScoreEncoder, AnswersEncoder, RollingScoreEncoder, RatioEncoder, UserAnswersEncoder
from riiid.core.transformers import ColumnsSelector, MemoryUsageLogger, RatioTransformer, WeightedAnswerTransformer, TypesTransformer, DensityTransformer
from riiid.core.featurers import SessionFeaturer, LecturesFeaturer, QuestionsFeaturer, LaggingFeaturer
from riiid.core.embedders import QuestionsEmbedder, AnswersCorrectnessEmbedder


class RiiidModel:

    def __init__(self, questions, lectures, params):
        self.questions = questions
        self.lectures = lectures
        self.params = params

        self.metadata = {}
        self.model_id: str = None
        self.lectures_pipeline: Pipeline = None
        self.pipeline: Pipeline = None
        self.features: t.List[str] = None
        self.model: lgb.LGBMModel = None
        self.best_score: float = None
        self.best_iteration: int = None

        self.test_batch: int = 0
        self.previous_test: pd.DataFrame = None

        self.context_encoders: t.List[RollingScoreEncoder] = None
        self.context_index: t.Dict[int, int] = None
        self.context_path: str = None
        self.context_loaded: t.Set[int] = None

    def get_name(self, prefix=None):
        if prefix:
            return 'model_{}_{}'.format(self.model_id, prefix)
        else:
            return 'model_{}.zip'.format(self.model_id)

    def get_normalized_name(self):
        return 'model_{}'.format(self.model_id).replace('_', '-')

    def _init_fit(self, X):
        self.model_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metadata['n_rows'] = len(X)
        self.metadata['n_users'] = X['user_id'].nunique()

    def make_lectures_pipeline(self):
        self.lectures_pipeline = make_pipeline(
            SessionFeaturer(hours_between_sessions=2),
            LecturesFeaturer(self.lectures),
            ColumnsSelector(columns_to_drop=['lecture_time', 'session_start']),
            TypesTransformer()
        )

    def make_global_pipeline(self, cv):

        self.pipeline = make_pipeline(
            ColumnsSelector(),

            QuestionsFeaturer(self.questions),
            UserAnswersEncoder(cv=cv, decay=0.99, smoothing_min=4, smoothing_value=1),
            AnswersEncoder(**self.params['answer_encoder']),

            QuestionsEmbedder(self.questions, **self.params['question_embedding']),
            AnswersCorrectnessEmbedder(**self.params['answers_embedding']),

            RatioEncoder('task_container_id'),
            RatioEncoder('question_category'),
            RatioEncoder('question_tag'),
            RatioEncoder('question_tags'),
            RatioEncoder('content_id'),

            #ScoreEncoder('task_container_id'),
            ScoreEncoder('question_part'),
            ScoreEncoder('question_category'),
            ScoreEncoder('question_tag'),
            #ScoreEncoder('question_tags', cv=cv, noise=0.001),
            ScoreEncoder('content_id', cv=cv, updatable=True, transformable=True, **self.params['score_encoder']),
            ScoreEncoder(['tasks_bucket_3', 'content_id'], cv=cv, parent_prior='content_id_encoded', updatable=True, **self.params['score_encoder_2']),
            ScoreEncoder(['tasks_bucket_12', 'content_id'], cv=cv, parent_prior='tasks_bucket_3_content_id_encoded', updatable=True, **self.params['score_encoder_2']),
            #ScoreEncoder(['prior_answered_correctly', 'content_id'], cv=cv, parent_prior='content_id', updatable=True, **self.params['score_encoder']),
            #ScoreEncoder('lecture_id', cv=cv, **self.params['score_encoder']),
            #ScoreEncoder('recent_lecture_id', cv=cv, **self.params['score_encoder']),

            WeightedAnswerTransformer('content_id_encoded'),

            RollingScoreEncoder(['user_id'], count=True, **self.params['user_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_part'], count=True, time_since_last=True, **self.params['user_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_category'], count=True, time_since_last=True, **self.params['user_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_tag'], count=True, time_since_last=True, **self.params['user_score_encoder']),

            RollingScoreEncoder(['user_id', 'content_id'], count=True, time_since_last=True, **self.params['user_content_score_encoder']),

            RollingScoreEncoder(['user_id'], **self.params['user_rolling_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_part'], **self.params['user_rolling_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_category'], **self.params['user_rolling_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_tag'], **self.params['user_rolling_score_encoder']),

            RollingScoreEncoder(['user_id'], weighted=True, **self.params['user_weighted_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_part'], weighted=True, **self.params['user_weighted_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_category'], weighted=True, **self.params['user_weighted_score_encoder']),
            RollingScoreEncoder(['user_id', 'question_tag'], weighted=True, **self.params['user_weighted_score_encoder']),

            RollingScoreEncoder(['user_id'], decay=0.99, smoothing_min=10, smoothing_value=2),
            RollingScoreEncoder(['user_id', 'question_part'], decay=0.99, smoothing_min=10, smoothing_value=2),

            RollingScoreEncoder(['user_id'], decay=0.9, smoothing_min=1, smoothing_value=1),
            RollingScoreEncoder(['user_id', 'question_part'], decay=0.9, smoothing_min=1, smoothing_value=1),
            RollingScoreEncoder(['user_id', 'question_category'], decay=0.9, smoothing_min=1, smoothing_value=1),
            RollingScoreEncoder(['user_id', 'question_tag'], decay=0.9, smoothing_min=1, smoothing_value=1),

            LaggingFeaturer(['content_id_encoded', 'mean_content_time', 'prior_question_time', 'prior_question_elapsed_time', 'prior_answer_ratio'], lag=3),
            # Removed 'task_diff', 'prior_question_had_explanation'

            DensityTransformer(['user_id_question_category_count', 'user_id_question_part_count', 'user_id_question_tag_count', 'user_id_content_id_count']),

            ColumnsSelector(columns_to_drop=[
                'user_id', 'content_id', 'user_answer', 'bundle_id', 'correct_answer',
                'answered_correctly', 'question_tags', 'answer_weight', 'lecture_id', 'lecture_tag', 'lecture_part',
                'tasks_bucket_3', 'tasks_bucket_12'
            ], validate=True),

            TypesTransformer(float_dtype=FLOAT_DTYPE),

            MemoryUsageLogger()
        )

    def fit_transform(self, X):
        self._init_fit(X)
        X = X.reset_index(drop=True)

        self.make_lectures_pipeline()
        X = self.lectures_pipeline.fit_transform(X)
        X = self.remove_lectures(X)

        train = pd.isnull(X['batch_id'])
        valid = ~pd.isnull(X['batch_id'])
        X = X.drop(columns=['batch_id'])
        y = X['answered_correctly']
        cv = self.build_cv(X, train, valid)

        self.make_global_pipeline(cv)
        X = self.pipeline.fit_transform(X, y)
        self.features = list(X.columns)

        DataFrameAnalyzer().summary(X)

        return X, y, train, valid

    @staticmethod
    def remove_lectures(X):
        # Remove lectures from train
        X = X[X['content_type_id'] == 0].reset_index(drop=True)
        X = X.drop(columns=['content_type_id'])
        return X

    def build_cv(self, X, train, valid):
        # We split the training part in folds, on create a new CV for validation with train = all train and test = test
        # We also pass True or False to apply noise to each fold. Validation fold is time-splitted, so it does not need noise
        Xtrain = X[train]
        cv = list(GroupKFold(n_splits=self.params['n_fold']).split(Xtrain, groups=Xtrain['user_id']))
        cv = [(Xtrain.iloc[cv_train].index.values, Xtrain.iloc[cv_valid].index.values, True) for cv_train, cv_valid in cv]
        cv.append((Xtrain.index.values, X[valid].index.values, False))
        return cv

    # Deprecated
    def get_validation_set(self, tests):
        X, y = [], []
        for i, test in enumerate(tests):
            if i % 500 == 0:
                logging.info('Computing validation set: {}/{}'.format(i, len(tests)))
            test = self.update(test)
            test = self.lectures_pipeline.transform(test)
            test = self.remove_lectures(test)
            if len(test) > 0:
                y.append(test['answered_correctly'])
                test = self.pipeline.transform(test)
                X.append(test)
        X = pd.concat(X)
        y = pd.concat(y)
        return X, y

    def fit_model(self, X, y, X_val, y_val):
        logging.info('- Fit lightgbm model')

        train_set = lgb.Dataset(X, y)
        val_set = lgb.Dataset(X_val, y_val)
        self.model = lgb.train(
            self.params['lgbm_params'],
            train_set,
            valid_sets=[val_set],
            num_boost_round=10000,
            early_stopping_rounds=50,
            verbose_eval=-1,
            callbacks=[logging_callback()]
        )
        self.best_score = self.model.best_score['valid_0']['auc']
        self.best_iteration = self.model.best_iteration
        logging.info('Best score = {:.2%}, in {} iterations'.format(self.best_score, self.best_iteration))

    def refit_model(self, X, y):
        logging.info('- Refit lightgbm model')

        train_set = lgb.Dataset(X, y)
        self.model = lgb.train(
            self.params['lgbm_params'],
            train_set,
            num_boost_round=self.best_iteration,
            verbose_eval=-1
        )

    def update(self, test):
        prior_user_answer = eval(test['prior_group_responses'].values[0])
        prior_answered_correctly = eval(test['prior_group_answers_correct'].values[0])
        test = test.drop(columns=['prior_group_answers_correct', 'prior_group_responses'])

        if self.previous_test is not None:
            self.previous_test['user_answer'] = prior_user_answer
            self.previous_test['answered_correctly'] = prior_answered_correctly

            X = self.previous_test
            X = update_pipeline(self.lectures_pipeline, X)
            X = self.remove_lectures(X)
            if len(X) > 0:
                y = X['answered_correctly']
                update_pipeline(self.pipeline, X, y)

        self.previous_test = test.copy()
        return test

    def predict(self, X):
        if self.test_batch % 1 == 0:
            logging.info('Running test batch {}'.format(self.test_batch))

        self.load_users(X['user_id'].unique())

        row_id = X['row_id']
        X = self.lectures_pipeline.transform(X)
        X['row_id'] = row_id
        X = self.remove_lectures(X)
        if len(X) > 0:
            predictions = X[['row_id']].copy()
            X = self.pipeline.transform(X)
            predictions['answered_correctly'] = self.model.predict(X)
        else:
            predictions = pd.DataFrame(columns=['row_id', 'answered_correctly'])
        self.test_batch += 1
        return X, predictions

    def save(self, path=None):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.writestr('context.txt', self.dumps_context())
            zip.writestr('model.pkl', pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

        if path is None:
            return zip_buffer
        else:
            with open(path, 'wb') as file:
                file.write(zip_buffer.getvalue())

    def save_with_source(self, path=None):
        zip_buffer = io.BytesIO()
        module_path = get_riiid_directory()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.writestr('model.zip', self.save().getvalue())
            for root, dirs, files in os.walk(module_path):
                for file in files:
                    arcname = os.path.join('riiid', os.path.relpath(os.path.join(root, file), module_path))
                    zip.write(os.path.join(root, file), arcname=arcname)

        if path is None:
            return zip_buffer
        else:
            with open(path, 'wb') as file:
                file.write(zip_buffer.getvalue())

    def dumps_context(self):
        buffer = io.StringIO()
        self.context_encoders = [e for _, e in (self.lectures_pipeline.steps + self.pipeline.steps) if hasattr(e, 'context')]

        # We chack that all context have the same users:
        for i, c in enumerate(self.context_encoders):
            if len(c.context) != len(self.context_encoders[0].context):
                raise ValueError('Contexts do not have the same size (context {})'.format(i))

        self.context_index = {}
        n = 0
        for user_id, _ in self.context_encoders[0].context.items():
            self.context_index[user_id] = n
            context = [encoder.get_user_context(user_id) for encoder in self.context_encoders]
            context = json.dumps(context)
            buffer.write(context)
            buffer.write('\n')
            n += len(context) + 1

        for encoder in self.context_encoders:
            encoder.context = {}

        return buffer.getvalue()

    def load_users(self, user_ids):
        for user_id in user_ids:
            user_id = int(user_id)
            if user_id not in self.context_loaded and user_id in self.context_index:
                self.context_loaded.add(user_id)
                contexts = self._get_user_from_cache(user_id)
                for encoder, context in zip(self.context_encoders, contexts):
                    encoder.set_user_context(user_id, context)

    def _get_user_from_cache(self, user_id):
        with open(self.context_path, 'r') as file:
            file.seek(self.context_index[user_id])
            line = file.readline()
            return json.loads(line, object_hook=keys_to_int)

    @staticmethod
    def load(path, working_path=None):
        if path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip:
                model = pickle.loads(zip.read('model.pkl'))
                if working_path is None:
                    working_path = str(Path(path).parent)
                model.context_path = os.path.join(working_path, 'context.txt')
                model.context_loaded = set()
                zip.extract('context.txt', working_path)
            return model
        else:
            model = load_pkl(os.path.join(path, 'model.pkl'))
            model.context_path = os.path.join(path, 'context.txt')
            model.context_loaded = set()
            return model
