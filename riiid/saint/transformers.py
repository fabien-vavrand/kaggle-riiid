import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer

from riiid.core.utils import tasks_bucket_12, indexed_merge, pre_filtered_indexed_merge
from riiid.core.computation import last_lecture


class LecturesTransformer:

    def __init__(self, lectures, time_bins=20):
        self.lectures = lectures
        self.time_bins = time_bins

        self.tasks_encoder = OrdinalEncoder(dtype=np.int8)
        self.time_discretizer = KBinsDiscretizer(n_bins=self.time_bins, encode='ordinal', strategy='quantile')
        self.features = None
        self.context = {}

    def fit(self, X, y=None):
        X['last_lecture_task_container_id'] = last_lecture(X, 'task_container_id')

        context = X[['user_id', 'timestamp', 'last_lecture_task_container_id']].drop_duplicates('user_id', keep='last')
        context.rename(columns={'timestamp': 'prior_timestamp'}, inplace=True)
        context = context.set_index('user_id')

        self.features = list(context.columns)
        self.context = context.to_dict(orient='index')
        return self

    def transform(self, X):
        X = self._merge_context(X)

        # Compute content time
        tasks = X.groupby('user_id').size()
        tasks.name = 'task_count'
        X = indexed_merge(X, tasks, left_on='user_id')
        X['content_time'] = (X['timestamp'] - X['prior_timestamp']) / X['task_count']
        X = X.drop(columns=['task_count'])
        not_nan = ~pd.isnull(X['content_time'])
        if not_nan.sum() > 0:
            X.loc[not_nan, 'content_time'] = self.time_discretizer.transform(X.loc[not_nan, ['content_time']]) + 1
        X['content_time'] = X['content_time'].fillna(0).astype(np.int8)

        # Compute tasks since last lecture
        X['tasks_since_last_lecture'] = X['task_container_id'] - X['last_lecture_task_container_id']
        X['tasks_since_last_lecture'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_12)
        X['tasks_since_last_lecture'] = self.tasks_encoder.transform(X[['tasks_since_last_lecture']]) + 1

        self._update_post_transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)

        # Compute content time
        tasks = X.groupby(['user_id', 'task_container_id'], sort=False).agg({'timestamp': ['min', 'count']})
        tasks.columns = ['content_time', 'task_count']
        tasks = tasks.reset_index()
        tasks['content_time'] = tasks.groupby('user_id')['content_time'].diff()
        tasks['content_time'] = tasks['content_time'] / tasks['task_count']
        tasks = tasks.drop(columns=['task_count'])
        tasks = tasks[~pd.isnull(tasks['content_time'])]
        tasks['content_time'] = self.time_discretizer.fit_transform(tasks[['content_time']]) + 1
        X = pd.merge(X, tasks, on=['user_id', 'task_container_id'], how='left')
        X['content_time'] = X['content_time'].fillna(0).astype(np.int8)

        # Compute tasks since last lecture
        X['tasks_since_last_lecture'] = X['task_container_id'] - X['last_lecture_task_container_id']
        X['tasks_since_last_lecture'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_12)
        X['tasks_since_last_lecture'] = self.tasks_encoder.fit_transform(X[['tasks_since_last_lecture']]) + 1

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
                    'last_lecture_task_container_id': np.nan if content_type_id[r] == 0 else int(task_container_id[r]),
                }

    def update(self, X, y=None):
        return self

    def update_transform(self, X, y=None):
        return X


class QuestionsTransformer:

    def __init__(self, questions):
        self.questions = questions.copy()

    def fit(self, X, y=None):
        def get_first_tag(x):
            if pd.isnull(x):
                return 0
            else:
                x = x.split(' ')
                return int(x[0])

        self.questions['tag'] = self.questions['tags'].apply(get_first_tag)
        self.questions['tag'] = OrdinalEncoder(dtype=np.int8).fit_transform(self.questions[['tag']]) + 1
        self.questions.rename(columns={'question_id': 'content_id'}, inplace=True)
        self.questions = self.questions[['content_id', 'part', 'tag']].set_index('content_id')
        return self

    def transform(self, X):
        X = pre_filtered_indexed_merge(X, self.questions, left_on='content_id')
        X['content_id'] += 1
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        X = pd.merge(X, self.questions, left_on='content_id', right_index=True, how='left')
        X = self._transform(X)
        return X

    def _transform(self, X):
        X['content_id_answered_correctly'] = X['content_id'] + X['answered_correctly'] * len(self.questions)
        X['content_id'] += 1
        X['answered_correctly'] += 1
        X['content_id_answered_correctly'] += 1
        return X

    def update_transform(self, X, y=None):
        # This line is required if we update all the batch N features at batch N+1
        #X = pre_filtered_indexed_merge(X, self.questions, left_on='content_id')
        X = self._transform(X)
        return X
