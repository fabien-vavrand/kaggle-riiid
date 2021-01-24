import logging
import itertools
import numpy as np
import pandas as pd

from riiid.config import FLOAT_DTYPE
from riiid.core.computation import rolling_sum, rolling_categories_count, last_lecture
from riiid.core.utils import indexed_merge, pre_filtered_indexed_merge, tasks_bucket_3, tasks_bucket_12, tasks_group


class SessionFeaturer:
    """
    Compute 5 features: 'session_id', 'session_time', 'content_time', 'prior_question_time', 'prior_question_lost_time'
    """

    def __init__(self, hours_between_sessions=2):
        self.hours_between_sessions = hours_between_sessions
        self.content_time_groupby = ['user_id', 'session_id']

        self.features = None
        self.context = {}

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def transform(self, X):
        self._update_pre_transform(X)

        X = self._merge_context(X)

        tasks = X.groupby('user_id').size()
        tasks.name = 'task_count'
        X = indexed_merge(X, tasks, left_on='user_id')

        X['session_time'] = X['timestamp'] - X['session_start']
        X['content_time'] = (X['timestamp'] - X['prior_timestamp']) / X['task_count']
        X = X.drop(columns=['task_count'])
        X['content_time'] = X['content_time'].replace({0: np.nan})
        X['lecture_time'] = (X['content_time'] * (X['content_type_id'] == 1)).replace({0: np.nan})
        X['prior_question_lost_time'] = X['prior_question_time'] - X['prior_question_elapsed_time']

        X['task_diff'] = X['task_container_id'] - X['prior_task_container_id']

        self._update_post_transform(X)
        return X

    def fit_transform(self, X, y=None):
        X = self._fit(X, y)
        X = self._downcast(X)
        return X

    def _fit(self, X, y=None):
        logging.info('- Fit sessions featurer')
        X = self._compute_session_id(X)
        X = self._compute_time_features(X)

        # Build context
        self.context = X[['user_id', 'timestamp', 'task_container_id', 'session_id', 'session_start']].drop_duplicates('user_id', keep='last')
        self.context.rename(columns={'timestamp': 'prior_timestamp', 'task_container_id': 'prior_task_container_id'}, inplace=True)
        questions_context = X[X['content_type_id'] == 0][['user_id', 'content_time']].drop_duplicates('user_id', keep='last')
        questions_context.rename(columns={'content_time': 'prior_question_time'}, inplace=True)
        self.context = pd.merge(self.context, questions_context, on='user_id', how='left').set_index('user_id')
        self.features = list(self.context.columns)
        self.context = self.context.to_dict(orient='index')
        return X

    def _downcast(self, X):
        X['prior_question_time'] = X['prior_question_time'].astype(FLOAT_DTYPE)
        X['prior_question_lost_time'] = X['prior_question_lost_time'].astype(FLOAT_DTYPE)
        X['task_diff'] = X['task_diff'].astype(FLOAT_DTYPE)
        return X

    def _compute_session_id(self, X):
        cache_id = 'session_featurer_session_id_{}'.format(len(X))
        user_id = -1
        sessions = np.zeros(len(X), dtype=np.int16)
        users = X['user_id'].values
        timestamps = X['timestamp'].values
        for r in range(len(X)):
            if user_id != users[r]:
                ts = 0
                session_id = 0
                user_id = users[r]
            if (timestamps[r] - ts) >= self.hours_between_sessions * 60 * 60 * 1000:
                session_id += 1
            sessions[r] = session_id
            ts = timestamps[r]

        X['session_id'] = sessions
        return X

    def _compute_time_features(self, X):
        # Compute sessions infos
        sessions = X.groupby(['user_id', 'session_id'])['timestamp'].min()
        sessions.name = 'session_start'

        # Merge sessions info with X
        X = pd.merge(X, sessions, how='left', on=['user_id', 'session_id'])
        X['session_time'] = X['timestamp'] - X['session_start']
        X['content_time'] = X.groupby(self.content_time_groupby)['timestamp'].diff()
        X['lecture_time'] = (X['content_time'] * (X['content_type_id'] == 1)).replace({0: np.nan})

        # Recalculate properly content_time for multi questions tasks
        tasks = X[X['content_type_id'] == 0].groupby(['user_id', 'session_id', 'task_container_id'], sort=False).agg({'content_time': [np.sum, 'count']})
        tasks.columns = ['content_time', 'task_count']
        tasks['content_time'] = tasks['content_time'] / tasks['task_count']
        tasks['content_time'] = tasks['content_time'].replace({0: np.nan})
        tasks = tasks.drop(columns=['task_count'])
        tasks['prior_question_time'] = tasks.groupby(self.content_time_groupby)['content_time'].shift()

        # Merge features with X
        X = X.drop(columns=['content_time'])
        X = pd.merge(X, tasks, on=['user_id', 'task_container_id'], how='left')
        X['prior_question_lost_time'] = X['prior_question_time'] - X['prior_question_elapsed_time']

        # Compute tasks evolution features
        tasks = X.groupby(['user_id', 'task_container_id'], sort=False).size().reset_index(name='task_diff')
        tasks['task_diff'] = tasks.groupby('user_id')['task_container_id'].diff()
        X = pd.merge(X, tasks, on=['user_id', 'task_container_id'], how='left')

        return X

    def _update_pre_transform(self, X):
        # Update session_id and session_start
        users = X['user_id'].values
        tss = X['timestamp'].values
        seen_users = set()
        for r in range(users.shape[0]):
            user_id = int(users[r])
            ts = int(tss[r])
            # We don't want to consider the same user 2 times (same task)
            if user_id in seen_users:
                continue
            seen_users.add(user_id)

            try:
                context = self.context[user_id]
                if ts - context['prior_timestamp'] >= self.hours_between_sessions * 60 * 60 * 1000:
                    context['prior_timestamp'] = np.nan  # So we are not able to compute content_time for the first element of a session
                    context['session_id'] += 1
                    context['session_start'] = ts
                    context['prior_question_time'] = np.nan
            except KeyError:
                # Warning: insertion order is IMPORTANT
                self.context[user_id] = {
                    'prior_timestamp': np.nan,
                    'prior_task_container_id': np.nan,
                    'session_id': 0,
                    'session_start': 0,
                    'prior_question_time': np.nan
                }

    def _update_post_transform(self, X):
        # Update prior_timestamp, prior_task_container_id, prior_question_time
        user_id = X['user_id'].values
        timestamp = X['timestamp'].values
        content_type_id = X['content_type_id'].values
        content_time = X['content_time'].values
        tasks = X['task_container_id'].values
        seen_users = set()
        for r in range(user_id.shape[0]):
            # We don't want to consider the same user 2 times (same task)
            if user_id[r] in seen_users:
                continue
            seen_users.add(user_id[r])

            context = self.context[user_id[r]]
            context['prior_timestamp'] = int(timestamp[r])
            context['prior_task_container_id'] = int(tasks[r])
            if content_type_id[r] == 0:
                context['prior_question_time'] = content_time[r]

    def _merge_context(self, X):
        users = X['user_id'].values
        results = np.empty((len(X), len(self.features)), dtype=np.float64)
        for r in range(len(X)):
            results[r, :] = list(self.context[users[r]].values())
        for i, feature in enumerate(self.features):
            X[feature] = results[:, i]
        return X

    def update_transform(self, X, y=None):
        return X

    def get_user_context(self, user_id):
        return list(self.context[user_id].values())

    def set_user_context(self, user_id, context):
        self.context[user_id] = {f: v for f, v in zip(self.features, context)}


class LecturesFeaturer:

    def __init__(self, lectures):
        self.lectures = lectures.set_index('lecture_id')
        self.features = None
        self.context = None

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        self._update_pre_transform(X)
        X = self._merge_context(X)
        X = self._transform(X)
        return X

    def fit_transform(self, X, y=None):
        X = self._fit(X)
        X = self._transform(X)
        X = self._downcast(X)
        return X

    def _fit(self, X):
        logging.info('- Fit lectures featurer')
        X['lecture_id'] = X['content_id'] * (X['content_type_id'] == 1)
        X = pd.merge(X, self.lectures, how='left', left_on='lecture_id', right_index=True)

        # Count lectures
        counts = rolling_sum(X[['user_id', 'content_type_id']].to_numpy())
        X['n_lectures'] = counts[:, 0]

        # Count lectures parts
        parts_features = [f'n_lectures_part{part}' for part in list(range(1, 8))]
        X['lecture_part'] = X['lecture_part'] - 1
        counts = rolling_categories_count(X[['user_id', 'lecture_part']].to_numpy(), n_categories=7)
        for i, feature in enumerate(parts_features):
            X[feature] = counts[:, i]

        # last lecture features
        last_lecture_features = ['tasks_since_last_lecture', 'time_since_last_lecture', 'last_lecture_time', 'lecture_id', 'lecture_tag', 'lecture_part']
        X['tasks_since_last_lecture'] = last_lecture(X, 'task_container_id')
        X['time_since_last_lecture'] = last_lecture(X, 'timestamp')
        X['last_lecture_time'] = last_lecture(X, 'lecture_time')
        X['lecture_id'] = last_lecture(X, 'lecture_id')
        X['lecture_tag'] = last_lecture(X, 'lecture_tag')
        X['lecture_part'] = last_lecture(X, 'lecture_part')
        X['type_of'] = last_lecture(X, 'type_of')

        X['lecture_id'] = X['lecture_id'].fillna(-1).astype(np.int16)
        X['lecture_tag'] = X['lecture_tag'].fillna(-1).astype(np.int16)
        X['lecture_part'] = X['lecture_part'].fillna(-1).astype(np.int8)
        X['type_of'] = X['type_of'].fillna(-1).astype(np.int8)

        # Build context
        self.features = ['n_lectures'] + parts_features + last_lecture_features
        self.context = X[['user_id'] + self.features].drop_duplicates('user_id', keep='last').set_index('user_id').to_dict(orient='index')
        return X

    def _transform(self, X):
        X['time_since_last_lecture'] = X['timestamp'] - X['time_since_last_lecture']
        X['tasks_since_last_lecture'] = X['task_container_id'] - X['tasks_since_last_lecture']
        X['tasks_bucket_12'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_12).astype(np.int16)
        X['tasks_bucket_3'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_3).astype(np.int8)
        return X

    def _downcast(self, X):
        X['last_lecture_time'] = X['last_lecture_time'].astype(FLOAT_DTYPE)
        X['time_since_last_lecture'] = X['time_since_last_lecture'].astype(FLOAT_DTYPE)
        X['tasks_since_last_lecture'] = X['tasks_since_last_lecture'].astype(FLOAT_DTYPE)
        return X

    def _update_pre_transform(self, X):
        X = X[X['content_type_id'] == 1]
        if len(X) == 0:
            return
        X = X.rename(columns={'content_id': 'lecture_id'})
        X = pre_filtered_indexed_merge(X, self.lectures, left_on='lecture_id')

        users = X['user_id'].values
        task_container_id = X['task_container_id'].values
        timestamp = X['timestamp'].values
        lecture_time = X['lecture_time'].values
        lecture_id = X['lecture_id'].values
        lecture_tag = X['lecture_tag'].values
        lecture_part = X['lecture_part'].values

        for r in range(len(users)):
            try:
                context = self.context[users[r]]
            except KeyError:
                context = {f: 0 for f in self.features}
                self.context[users[r]] = context

            context['n_lectures'] += 1
            context[f'n_lectures_part{lecture_part[r]}'] += 1
            context['tasks_since_last_lecture'] = int(task_container_id[r])
            context['time_since_last_lecture'] = int(timestamp[r])
            context['last_lecture_time'] = float(lecture_time[r])
            context['lecture_id'] = int(lecture_id[r])
            context['lecture_tag'] = int(lecture_tag[r])
            context['lecture_part'] = int(lecture_part[r])

    def _merge_context(self, X):
        users = X['user_id'].values
        results = np.zeros((len(X), len(self.features)), dtype=np.float64)
        findex = {f: i for i, f in enumerate(self.features)}

        for r in range(len(X)):
            try:
                results[r, :] = list(self.context[users[r]].values())
            except KeyError:
                results[r, findex['tasks_since_last_lecture']] = np.nan
                results[r, findex['time_since_last_lecture']] = np.nan
                results[r, findex['last_lecture_time']] = np.nan
                results[r, findex['lecture_id']] = -1
                results[r, findex['lecture_tag']] = -1
                results[r, findex['lecture_part']] = -1

        for i, feature in enumerate(self.features):
            X[feature] = results[:, i]
        return X

    def update_transform(self, X, y=None):
        users = X['user_id'].values
        lecture_id = np.zeros(len(X), dtype=np.int16)
        tasks_since_last_lecture = np.zeros(len(X))
        for r in range(len(X)):
            try:
                context = self.context[users[r]]
                lecture_id[r] = context['lecture_id']
                tasks_since_last_lecture[r] = context['tasks_since_last_lecture']
            except KeyError:
                lecture_id[r] = -1
                tasks_since_last_lecture[r] = np.nan
        X['lecture_id'] = lecture_id
        X['tasks_since_last_lecture'] = tasks_since_last_lecture
        X['tasks_since_last_lecture'] = X['task_container_id'] - X['tasks_since_last_lecture']
        X['tasks_bucket_12'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_12)
        X['tasks_bucket_3'] = X['tasks_since_last_lecture'].fillna(0).apply(tasks_bucket_3)
        return X

    def get_user_context(self, user_id):
        return list(self.context[user_id].values())

    def set_user_context(self, user_id, context):
        self.context[user_id] = {f: v for f, v in zip(self.features, context)}


class QuestionsFeaturer:

    def __init__(self, questions):
        self.questions = questions.set_index('content_id')
        self.content_time = None

    def fit(self, X, y=None):
        logging.info('- Fit questions featurer')

        # Compute mean_content_time
        self.content_time = X['content_time'].median()
        questions = X[['content_id', 'content_time']].dropna().groupby('content_id')['content_time'].agg(['median', 'count'])
        questions.columns = ['mean_content_time', 'n_questions']
        self.questions = pd.merge(self.questions, questions, on='content_id', how='left')
        self.questions['mean_content_time'] = self.questions['mean_content_time'].fillna(self.content_time)
        self.questions['n_questions'] = self.questions['n_questions'].fillna(0)
        weight = (1 / (1 + np.exp(-(self.questions['n_questions'] - 5) / 1))) * (self.questions['n_questions'] > 0)
        self.questions['mean_content_time'] = weight * self.questions['mean_content_time'] + (1 - weight) * self.content_time
        self.questions = self.questions.drop(columns=['n_questions'])

        # Keep list of users for whom the first question was 7900
        self.users_7900 = set(X[(X['task_container_id'] == 0) & (X['content_id'] == 7900)]['user_id'].unique())

        return self

    def transform(self, X):
        self._update_pre_transform(X)

        X = pre_filtered_indexed_merge(X, self.questions, left_on='content_id')
        X['mean_content_time'] = (X['content_time'] / X['mean_content_time']).astype(FLOAT_DTYPE)
        X['content_time'] = X['content_time'].astype(FLOAT_DTYPE)
        X['prior_question_had_explanation'] = ((X['prior_question_had_explanation'] * 1).fillna(0)).astype(np.int8)

        # Retrieve lecture part count for question part, and delete lectures part counts
        X['n_lectures_on_question_part'] = self._get_n_lectures_on_question_part(X)
        X = X.drop(columns=[f'n_lectures_part{part}' for part in list(range(1, 8))])

        # compute task size
        tasks = X.groupby(['user_id', 'task_container_id']).size()
        tasks.name = 'task_size'
        X = indexed_merge(X, tasks, left_on=['user_id', 'task_container_id'])
        X['task_size'] = X['task_size'].astype(np.int8)

        X['user_7900'] = (X['user_id'].isin(self.users_7900) * 1).astype(np.int8)

        return X

    @staticmethod
    def _get_n_lectures_on_question_part(X):
        results = np.zeros(len(X), dtype=np.int16)
        for i, r in enumerate(X[['question_part'] + [f'n_lectures_part{p}' for p in range(1, 8)]].itertuples(index=False, name=None)):
            results[i] = r[r[0]]  # r[0] is question part, so r[r[0]] is the rolling count of lecture on question part
        return results

    @staticmethod
    def _get_n_lectures_on_question_tag(X):
        results = np.zeros(len(X), dtype=np.int16)
        unique_tags = [tag for tag in sorted(X['question_tag'].unique()) if f"n_lectures_tag{tag}" in X.columns]
        tags_features = [f"n_lectures_tag{tag}" for tag in unique_tags]
        mapping = {tag: i for i, tag in enumerate(unique_tags)}

        tags = X['question_tag'].values
        lectures_tags = X[tags_features].values
        for r in range(len(tags)):
            if tags[r] in mapping:
                results[r] = lectures_tags[r, mapping[tags[r]]]
        return results

    def _update_pre_transform(self, X, y=None):
        user_id = X['user_id'].values
        content_id = X['content_id'].values
        task_container_id = X['task_container_id'].values

        for r in range(len(user_id)):
            if task_container_id[r] == 0 and content_id[r] == 7900:
                self.users_7900.add(user_id[r])

    def update_transform(self, X, y=None):
        X = pre_filtered_indexed_merge(X, self.questions, left_on='content_id')
        X['user_7900'] = (X['user_id'].isin(self.users_7900) * 1).astype(np.int8)
        return X


class LaggingFeaturer:

    def __init__(self, features, lag=1):
        self.features = features
        self.lag = lag

        self.lags = [l + 1 for l in range(self.lag)]
        self.feature_names = [column + '_lag_' for column in self.features]
        self.dtypes = {}
        self.context = {}

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        X = self._merge_context(X)
        self._update_post_transform(X)
        return X

    def fit_transform(self, X, y=None):
        X = self._fit(X)
        return X

    def _fit(self, X):
        logging.info('- Fit lag features')
        self.dtypes = {f: X[f].dtype for f in self.features}
        tasks = X.groupby(['user_id', 'task_container_id'], sort=False)[self.features].agg(np.mean).reset_index()
        self._build_context(tasks)

        for l in self.lags:
            tasks[self.features] = tasks.groupby('user_id')[self.features].shift()
            lagged_features = [f + str(l) for f in self.feature_names]
            tasks_lagged = tasks.rename(columns={f: lf for f, lf in zip(self.features, lagged_features)})
            X = pd.merge(X, tasks_lagged, on=['user_id', 'task_container_id'], how='left')
        return X

    def _build_context(self, X):
        X = X.groupby('user_id').tail(self.lag)
        user_id = X['user_id'].values
        values = {f: X[f].values for f in self.features}

        for r in range(len(user_id)):
            if user_id[r] not in self.context:
                self.context[user_id[r]] = {f: [] for f in self.features}
            context = self.context[user_id[r]]
            for f in self.features:
                context[f].append(self._cast_value(f, values[f][r]))

    def _cast_value(self, feature, value):
        if pd.api.types.is_integer_dtype(self.dtypes[feature]):
            return int(value)
        else:
            return float(value)

    def _merge_context(self, X):
        user_id = X['user_id'].values
        n_features = self.lag * len(self.features)
        results = np.empty((len(X), n_features), dtype=np.float64)
        for r in range(len(X)):
            try:
                context = self.context[user_id[r]]
                for c, (lag, column) in enumerate(itertools.product(self.lags, self.features)):
                    try:
                        results[r, c] = context[column][-lag]
                    except:
                        results[r, c] = np.nan
            except:
                results[r, :] = [np.nan] * n_features

        for c, (lag, feature) in enumerate(itertools.product(self.lags, self.feature_names)):
            X[feature + str(lag)] = results[:, c]
        return X

    def _update_post_transform(self, X):
        X = X.groupby(['user_id', 'task_container_id'], sort=False)[self.features].agg(np.mean).reset_index()
        user_id = X['user_id'].values
        values = {f: X[f].values for f in self.features}
        for r in range(len(X)):
            try:
                context = self.context[user_id[r]]
            except:
                context = {f: [] for f in self.features}
                self.context[user_id[r]] = context

            for f in self.features:
                if len(context[f]) >= self.lag:
                    context[f].pop(0)
                context[f].append(self._cast_value(f, values[f][r]))

    def update_transform(self, X, y=None):
        return X

    def get_user_context(self, user_id):
        return list(self.context[user_id].values())

    def set_user_context(self, user_id, context):
        self.context[user_id] = {f: v for f, v in zip(self.features, context)}
