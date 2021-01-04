import logging
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, StandardScaler

from riiid.core.utils import tasks_bucket_12, indexed_merge, pre_filtered_indexed_merge
from riiid.core.computation import last_lecture


class LecturesTransformer:

    def __init__(self, lectures):
        self.lectures = lectures

        self.tasks_encoder = OrdinalEncoder(dtype=np.int8)

        self.features = None
        self.context = {}

    def fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X):
        X = self._merge_context(X)

        # Compute content time
        tasks = X.groupby('user_id').size()
        tasks.name = 'task_count'
        X = indexed_merge(X, tasks, left_on='user_id')
        X['content_time'] = (X['timestamp'] - X['prior_timestamp']) / X['task_count']

        # Compute tasks since last lecture
        X['tasks_since_last_lecture'] = X['task_container_id'] - X['last_lecture_task_container_id']
        X['tasks_since_last_lecture'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_12)
        X['tasks_since_last_lecture'] = self.tasks_encoder.transform(X[['tasks_since_last_lecture']]) + 1

        self._update_post_transform(X)
        return X

    def fit_transform(self, X, y=None):
        # Compute content time
        tasks = X.groupby(['user_id', 'task_container_id'], sort=False).agg({'timestamp': ['min', 'count']})
        tasks.columns = ['content_time', 'task_count']
        tasks = tasks.reset_index()
        tasks['content_time'] = tasks.groupby('user_id')['content_time'].diff()
        tasks['content_time'] = tasks['content_time'] / tasks['task_count']
        tasks = tasks.drop(columns=['task_count'])
        X = pd.merge(X, tasks, on=['user_id', 'task_container_id'], how='left')

        # Compute tasks since last lecture
        X['last_lecture_task_container_id'] = last_lecture(X, 'task_container_id')
        X['tasks_since_last_lecture'] = X['task_container_id'] - X['last_lecture_task_container_id']
        X['tasks_since_last_lecture'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_12)
        X['tasks_since_last_lecture'] = self.tasks_encoder.fit_transform(X[['tasks_since_last_lecture']]) + 1

        # Build context
        context = X[['user_id', 'timestamp', 'last_lecture_task_container_id']].drop_duplicates('user_id', keep='last')
        context.rename(columns={'timestamp': 'prior_timestamp'}, inplace=True)
        context = context.set_index('user_id')

        self.features = list(context.columns)
        self.context = context.to_dict(orient='index')

        return X

    def _merge_context(self, X):
        users = X['user_id'].values
        results = np.full((len(X), len(self.features)), np.nan, dtype=np.float64)
        for r in range(len(X)):
            try:
                results[r, :] = list(self.context[users[r]].values())
            except KeyError:
                pass  # Default values to nan

        for i, feature in enumerate(self.features):
            X[feature] = results[:, i]
        return X

    def _update_post_transform(self, X):
        # Update prior_timestamp, last_lecture_task_container_id
        user_id = X['user_id'].values
        timestamp = X['timestamp'].values
        content_type_id = X['content_type_id'].values
        task_container_id = X['task_container_id'].values
        seen_users = set()
        for r in range(user_id.shape[0]):
            # We don't want to consider the same user 2 times (same task)
            if user_id[r] in seen_users:
                continue
            seen_users.add(user_id[r])

            try:
                context = self.context[user_id[r]]
                context['prior_timestamp'] = int(timestamp[r])
                if content_type_id[r] == 1:
                    context['last_lecture_task_container_id'] = int(task_container_id[r])
            except KeyError:
                self.context[user_id[r]] = {
                    'prior_timestamp': int(timestamp[r]),
                    'last_lecture_task_container_id': np.nan if content_type_id[r] == 0 else int(task_container_id[r])
                }

    def update(self, X, y=None):
        return self

    def update_transform(self, X, y=None):
        return X


class QuestionsTransformer:

    def __init__(self, questions, tags_dimension=64, time_bins=20, elapsed_bins=20, lag_bins=20):
        self.questions = questions.copy()
        self.tags_dimension = tags_dimension

        self.time_bins = time_bins
        self.elapsed_bins = elapsed_bins
        self.lag_bins = lag_bins

        self.time_discretizer = KBinsDiscretizer(n_bins=self.time_bins, encode='ordinal', strategy='quantile')
        self.elapsed_scaler = StandardScaler()
        self.lag_discretizer = KBinsDiscretizer(n_bins=self.lag_bins, encode='ordinal', strategy='quantile')

        self.features = None
        self.context = {}

    def fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X):
        X = pre_filtered_indexed_merge(X, self.questions, left_on='content_id')
        X['content_id'] += 1

        X = self._merge_context(X)
        X['prior_question_lag_time'] = X['prior_content_time'] - X['prior_question_elapsed_time']

        # Rename prior features
        X.rename(columns={'prior_question_elapsed_time': 'question_elapsed_time',
                          'prior_question_lag_time': 'question_lag_time',
                          'prior_question_had_explanation': 'question_had_explanation'}, inplace=True)

        # Update is performed before encoding to update prior_content_time
        self._update_post_transform(X)

        # Transform
        X = self._transform(X)
        return X

    def fit_transform(self, X, y=None):
        # Fit and merge questions
        X = self._fit_questions(X)

        # Compute prior_content_time
        tasks = X[['user_id', 'task_container_id', 'content_time']].drop_duplicates()
        tasks['prior_content_time'] = tasks.groupby('user_id')['content_time'].shift()
        X = pd.merge(X, tasks[['user_id', 'task_container_id', 'prior_content_time']], on=['user_id', 'task_container_id'], how='left')
        X['prior_question_lag_time'] = X['prior_content_time'] - X['prior_question_elapsed_time']

        # Shift priors
        prior_features = ['prior_question_elapsed_time', 'prior_question_lag_time', 'prior_question_had_explanation']
        tasks = X[['user_id', 'task_container_id'] + prior_features].drop_duplicates(['user_id', 'task_container_id'])
        tasks[prior_features] = tasks.groupby('user_id')[prior_features].shift(-1)
        tasks.rename(columns=lambda x: x.replace('prior_', ''), inplace=True)
        X = X.drop(columns=prior_features)
        X = pd.merge(X, tasks, on=['user_id', 'task_container_id'], how='left')

        # Encode
        self.time_discretizer.fit(X.loc[~pd.isnull(X['content_time']), ['content_time']])
        self.elapsed_scaler.fit(np.log(X.loc[~pd.isnull(X['question_elapsed_time']), ['question_elapsed_time']] / 1000 + 1))
        self.lag_discretizer.fit(X.loc[~pd.isnull(X['question_lag_time']), ['question_lag_time']])

        # Build context
        context = X[['user_id', 'content_time']].drop_duplicates('user_id', keep='last')
        context.rename(columns={'content_time': 'prior_content_time'}, inplace=True)
        context = context.set_index('user_id')

        self.features = list(context.columns)
        self.context = context.to_dict(orient='index')

        # Transform
        X = self._transform(X)
        X = self._transform_shared(X)

        return X

    def _fit_questions(self, X):
        self.questions['tags'] = self.questions['tags'].fillna('0')
        self.questions['tag'] = self.questions['tags'].apply(lambda x: int(x.split(' ')[0]))
        self.questions['tag'] = OrdinalEncoder(dtype=np.int8).fit_transform(self.questions[['tag']]) + 1
        self.questions['tags'] = OrdinalEncoder(dtype=np.int16).fit_transform(self.questions[['tags']]) + 1
        self.questions.rename(columns={'question_id': 'content_id'}, inplace=True)
        self.questions = self.questions[['content_id', 'part', 'tags']].set_index('content_id')
        X = pd.merge(X, self.questions, left_on='content_id', right_index=True, how='left')
        return X

    def _reduce_tags_dimensions(self):
        # Deprecated: it could be used to initialize the tags embedding, but I will just let the model learn that by itself
        tags = set()
        questions_tags = self.questions['tags'].values
        for r in range(len(questions_tags)):
            tags.update(questions_tags[r].split(' '))
        tags = sorted(list(map(int, tags)))
        logging.info('Found {} tags, from {} to {}'.format(len(tags), tags[0], tags[-1]))

        results = np.zeros((len(self.questions), len(tags)), dtype=np.float32)
        for r in range(len(questions_tags)):
            for c in list(map(int, questions_tags[r].split(' '))):
                results[r, c] = 1

        self.tags_reduction = PCA(n_components=self.tags_dimension).fit_transform(results)

    def _transform(self, X):
        not_nan = ~pd.isnull(X['content_time'])
        if not_nan.sum() > 0:
            X.loc[not_nan, 'content_time'] = self.time_discretizer.transform(X.loc[not_nan, ['content_time']]) + 1
        X['content_time'] = X['content_time'].fillna(0).astype(np.int8)

        not_nan = ~pd.isnull(X['question_elapsed_time'])
        if not_nan.sum() > 0:
            X.loc[not_nan, 'question_elapsed_time'] = self.elapsed_scaler.transform(np.log(X.loc[not_nan, ['question_elapsed_time']] / 1000 + 1))
        X['question_elapsed_time'] = X['question_elapsed_time'].fillna(0).astype(np.float32)

        not_nan = ~pd.isnull(X['question_lag_time'])
        if not_nan.sum() > 0:
            X.loc[not_nan, 'question_lag_time'] = self.lag_discretizer.transform(X.loc[not_nan, ['question_lag_time']]) + 1
        X['question_lag_time'] = X['question_lag_time'].fillna(0).astype(np.int8)

        X['question_had_explanation'] += 1
        X['question_had_explanation'] = X['question_had_explanation'].fillna(0).astype(np.int8)

        return X

    def _transform_shared(self, X):
        X['content_id_answered_correctly'] = X['content_id'] + X['answered_correctly'] * len(self.questions)
        X['content_id'] += 1
        X['answered_correctly'] += 1
        X['content_id_answered_correctly'] += 1
        return X

    def _merge_context(self, X):
        users = X['user_id'].values
        results = np.full((len(X), len(self.features)), np.nan, dtype=np.float64)
        for r in range(len(X)):
            try:
                results[r, :] = list(self.context[users[r]].values())
            except KeyError:
                pass  # Default values to nan

        for i, feature in enumerate(self.features):
            X[feature] = results[:, i]
        return X

    def _update_post_transform(self, X):
        # Update prior_content_time
        user_id = X['user_id'].values
        content_time = X['content_time'].values
        seen_users = set()
        for r in range(user_id.shape[0]):
            # We don't want to consider the same user 2 times (same task)
            if user_id[r] in seen_users:
                continue
            seen_users.add(user_id[r])

            try:
                context = self.context[user_id[r]]
                context['prior_content_time'] = float(content_time[r])
            except KeyError:
                self.context[user_id[r]] = {
                    'prior_content_time': float(content_time[r])
                }

    def update_transform(self, X, y=None):
        # This line is required if we update all the batch N features at batch N+1
        #X = pre_filtered_indexed_merge(X, self.questions, left_on='content_id')
        X = self._transform_shared(X)
        return X
