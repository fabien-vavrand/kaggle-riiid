import logging
import typing as t
import numpy as np
import pandas as pd

from sklearn.utils import check_random_state

from riiid.config import FLOAT_DTYPE
from riiid.core.computation import rolling_score, sorted_rolling_score, last_feature_value_time, compute_user_answers_ratio


np.seterr(divide='ignore', invalid='ignore')


class Smoother:

    def __init__(self, smoothing_min, smoothing_value):
        self.smoothing_min = smoothing_min
        self.smoothing_value = smoothing_value
        self.is_smooting_activated = self.smoothing_min is not None and self.smoothing_value is not None

    def smooth(self, nb):
        if self.smoothing_value == 0:
            return 1.0
        return 1 / (1 + np.exp(-(nb - self.smoothing_min) / self.smoothing_value))


class ScoreEncoder(Smoother):

    def __init__(
        self, columns, parent_prior=None, cv=None,
            smoothing_min=5, smoothing_value=1, noise=None,
            updatable=False, transformable=False
    ):
        super().__init__(smoothing_min, smoothing_value)
        self.columns = columns
        self.parent_prior = parent_prior
        self.cv = cv
        self.noise = noise
        self.updatable = updatable
        self.transformable = transformable

        self.columns_name = '_'.join(self.columns if isinstance(self.columns, list) else [self.columns])
        self.target_name = '{}_encoded'.format(self.columns_name)
        self.target_columns = ['weight', self.target_name]
        self.random_gen = check_random_state(123)

        self.prior = None
        self.posteriors = {}

    def fit(self, X, y=None):
        logging.info('- Fit target encoding for {}'.format(self.columns_name))
        self.prior, self.posteriors = self._fit(X)
        self.posteriors = self.posteriors.to_dict(orient='index')
        return self

    def transform(self, X):
        X = self._merge_context(X)
        X[self.target_name] = X['weight'] * X[self.target_name] + (1 - X['weight']) * self._get_prior(X, self.prior)
        X = X.drop(columns=['weight'])
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        if self.cv is None:
            prior, posteriors = self._fit(X)
            results = self._transform(X, prior, posteriors, noise=self.noise)
        else:
            results = []
            for fold, (train, test, noise) in enumerate(self.cv):
                prior, posteriors = self._fit(X.loc[train])
                sub_results = self._transform(X.loc[test], prior, posteriors, noise=self.noise if noise else None)
                results.append(sub_results)

            results = pd.concat(results, axis=0)
            results = results.loc[X.index, :]
            self.cv = None

        results[self.target_name] = results[self.target_name].astype(FLOAT_DTYPE)
        return results

    def _fit(self, X):
        prior = X['answered_correctly'].mean()
        posteriors = X.groupby(self.columns)['answered_correctly'].agg([np.sum, 'count'])
        posteriors['sum'] = posteriors['sum'].astype(np.int64)
        posteriors = self._process_posteriors(posteriors)
        return prior, posteriors

    def _process_posteriors(self, posteriors):
        posteriors['weight'] = self.smooth(posteriors['count'])
        posteriors[self.target_name] = posteriors['sum'] / posteriors['count']
        return posteriors

    def _transform(self, X, prior, posteriors, noise=None):
        X = pd.merge(X, posteriors[self.target_columns], on=self.columns, how='left').set_index(X.index)
        X['weight'] = X['weight'].fillna(0)
        X[self.target_name] = X[self.target_name].fillna(prior)   # Fill anything won't change the result as weight = 0
        X[self.target_name] = X['weight'] * X[self.target_name] + (1 - X['weight']) * self._get_prior(X, prior)
        X = X.drop(columns=['weight'])

        if noise is not None:
            X[self.target_name] += (self.random_gen.rand(len(X)) - 0.5) * noise
            X[self.target_name] = np.minimum(np.maximum(X[self.target_name], 0), 1)
        return X

    def _get_prior(self, X, prior):
        if self.parent_prior is not None:
            return X[self.parent_prior]
        else:
            return prior

    def _merge_context(self, X):
        values = X[self.columns].values
        rows = values.shape[0]
        results = np.zeros((rows, len(self.target_columns)), dtype=np.float64)
        for r in range(rows):
            try:
                results[r, :] = self._context_vector(values[r])
            except KeyError:
                pass

        for i, feature in enumerate(self.target_columns):
            X[feature] = results[:, i]
        return X

    def _context_vector(self, col_id):
        col_id = self._get_column_id(col_id)
        context = self.posteriors[col_id]
        return [context[column] for column in self.target_columns]

    def _get_column_id(self, col_id):
        if isinstance(col_id, int) or isinstance(col_id, tuple):
            return col_id
        elif isinstance(col_id, np.ndarray):
            return tuple(map(int, col_id))
        else:
            return int(col_id)

    def update(self, X, y=None):
        values = X[self.columns].values
        target = y.values
        for r in range(values.shape[0]):
            col_id = self._get_column_id(values[r])
            try:
                posterior = self.posteriors[col_id]
                posterior['sum'] += int(target[r])
                posterior['count'] += 1
            except KeyError:
                posterior = {
                    'sum': int(target[r]),
                    'count': 1
                }
                self.posteriors[col_id] = posterior
            self._process_posteriors(posterior)

    def update_transform(self, X, y=None):
        if self.updatable:
            self.update(X, y)
        if self.transformable:
            X = self.transform(X)
        return X


class RollingScoreEncoder(Smoother):

    def __init__(self, columns, rolling=None, smoothing_min=5, smoothing_value=1, count=False, weighted=False, decay=None, time_since_last=False):
        super().__init__(smoothing_min, smoothing_value)
        self.columns = columns
        self.rolling = rolling
        self.count = count
        self.weighted = weighted
        self.decay = decay
        self.time_since_last = time_since_last

        if self.rolling and self.time_since_last:
            raise ValueError('Not allowed to compute time and tasks when rolling')

        if self.rolling and self.decay is not None:
            raise ValueError('Not allowed to use rolling and decay at the same time')

        if self.time_since_last and len(self.columns) != 2:
            raise ValueError('Expecting 2 columns when computing time or tasks')

        rolling_name = '_{}'.format(self.rolling) if self.rolling else ''
        decay_name = '_decay_{}'.format(self.decay) if self.decay is not None else ''
        weighted_name = '_weighted' if self.weighted else ''
        self.name = '_'.join(self.columns) + weighted_name + '_score' + rolling_name + decay_name
        self.name_count = '_'.join(self.columns) + weighted_name + '_count' + rolling_name + decay_name
        self.compute_dtype = np.float64 if self.weighted or self.decay else np.int16
        if self.time_since_last:
            self.time_name = 'time_since_last_{}'.format(self.columns[-1])
            self.tasks_name = 'tasks_since_last_{}'.format(self.columns[-1])

        self.prior = None
        self.context = {}

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def transform(self, X):
        X = self._merge_context(X)
        return X

    def fit_transform(self, X, y=None):
        X = self._fit(X, y)
        X = self._transform(X)
        X = self._downcast(X)
        return X

    def _fit(self, X, y):
        logging.info('- Fit rolling score encoder for columns {}, rolling {}, weighted {}, decay {}'.format(self.columns, self.rolling, self.weighted, self.decay))
        self.global_score = y.mean()
        if self.columns[-1] + '_encoded' in X.columns:
            self.prior = self.columns[-1] + '_encoded'
        X = self._compute_rolling_scores(X)

        if self.time_since_last:
            X = self._compute_last_time_feature(X, self.time_name, 'timestamp')
            X = self._compute_last_time_feature(X, self.tasks_name, 'task_container_id')

        return X

    def _compute_last_time_feature(self, X, feature_name, feature):
        X[feature_name] = last_feature_value_time(X, feature, self.columns[-1])

        # Add to context
        context = X[['user_id', self.columns[-1], feature]].drop_duplicates(['user_id', self.columns[-1]], keep='last')
        user_id = context['user_id'].values
        column_values = context[self.columns[-1]].values
        feature_values = context[feature].values

        for r in range(len(user_id)):
            uid = int(user_id[r])
            cv = int(column_values[r])
            fv = int(feature_values[r])
            self.context[uid][cv].append(fv)

        return X

    def _compute_rolling_scores(self, X):
        data = X[self.columns + ['timestamp', 'task_container_id', 'answered_correctly']].values
        answer_weights = X['answer_weight'].values
        if len(self.columns) == 1 and self.columns[0] == 'user_id':
            results, keys, values, weights, contexts = sorted_rolling_score(data, answer_weights, rolling=self.rolling, weighted=self.weighted, decay=self.decay, dtype=self.compute_dtype)
        else:
            results, keys, values, weights, contexts = rolling_score(data, answer_weights, rolling=self.rolling, weighted=self.weighted, decay=self.decay, dtype=self.compute_dtype)

        X[self.name] = results[:, 0]
        X[self.name_count] = results[:, 1]

        for key, roll, weight, context in zip(keys, values, weights, contexts):
            self._add_to_context(key, roll, weight, context)
        return X

    def _add_to_context(self, key, roll, weight, key_context):
        key = list(map(int, key))  # Convert np.int to int to save memory
        context = self.context
        for k in key[:-1]:
            try:
                context = context[k]
            except KeyError:
                context[k] = {}
                context = context[k]
        new_context = self._context_from_rolls(roll, weight, key_context)
        context[key[-1]] = new_context
        return new_context

    def _context_from_rolls(self, roll, weight, key_context):
        if self.rolling:
            if self.weighted:
                return [roll, weight]
            else:
                return roll
        else:
            if self.weighted or self.decay is not None:
                return key_context
            else:
                return list(map(int, key_context))

    def _transform(self, X):
        if self.is_smooting_activated:
            X[self.name] = (X[self.name] / X[self.name_count]).fillna(0)
            weight = self.smooth(X[self.name_count]) * (X[self.name_count] > 0)
            X[self.name] = weight * X[self.name] + (1 - weight) * self._get_prior(X)
        else:
            X[self.name] = X[self.name] / X[self.name_count]

        if not self.count:
            X = X.drop(columns=[self.name_count])

        if self.time_since_last:
            X[self.time_name] = X['timestamp'] - X[self.time_name]
            X[self.tasks_name] = X['task_container_id'] - X[self.tasks_name]
        return X

    def _downcast(self, X):
        columns = [self.name, self.count]
        if self.time_since_last:
            columns += [self.time_name, self.tasks_name]
        for column in columns:
            if column in X.columns and pd.api.types.is_float_dtype(X[column].dtype) and X[column].dtype != FLOAT_DTYPE:
                X[column] = X[column].astype(FLOAT_DTYPE)
        return X

    def _get_prior(self, X):
        if self.prior:
            return X[self.prior].values
        else:
            return self.global_score

    def _context_vector(self, context):
        if self.rolling:
            if self.weighted:
                return [float(np.sum([r * w for r, w in zip(context[0], context[1])])), float(np.sum(context[1]))]
            else:
                return [int(np.sum(context)), len(context)]
        else:
            return context

    def _merge_context(self, X):
        values = [X[column].values for column in self.columns]
        rows, columns = len(X), len(values)
        results = np.zeros((rows, 2), dtype=self.compute_dtype)
        if self.time_since_last:
            time_results = np.full((rows, 2), np.nan)
        for r in range(rows):
            try:
                context = self.context
                for c in range(columns):
                    context = context[values[c][r]]
                context = self._context_vector(context)
                results[r, :] = context[:2]
                if self.time_since_last:
                    time_results[r, :] = context[2:]
            except KeyError:
                pass

        scores = results[:, 0] / results[:, 1]
        if self.is_smooting_activated:
            np.nan_to_num(scores, copy=False)
            weight = self.smooth(results[:, 1]) * (results[:, 1] > 0)
            X[self.name] = weight * scores + (1 - weight) * self._get_prior(X)
        else:
            X[self.name] = scores
        if self.count:
            X[self.name_count] = results[:, 1]
        if self.time_since_last:
            X[self.time_name] = time_results[:, 0]
            X[self.time_name] = X['timestamp'] - X[self.time_name]
            X[self.tasks_name] = time_results[:, 1]
            X[self.tasks_name] = X['task_container_id'] - X[self.tasks_name]
        return X

    def update(self, X, y=None):
        # values = X[self.columns].values
        # Trick to speed up the .values
        values = [X[column].values for column in self.columns]
        target = y.values
        weights = X['answer_weight'].values
        if self.time_since_last:
            timestamp = X['timestamp'].values
            task_container_id = X['task_container_id'].values
        rows, columns = len(X), len(values)
        decayed = set()  # As we don't want to apply decay multiple times for a multi questions task
        for r in range(rows):
            try:
                context = self.context
                for c in range(columns):
                    context = context[values[c][r]]
                if self.rolling:
                    if self.weighted:
                        context[0].append(int(target[r]))
                        context[1].append(weights[r])
                        if len(context[0]) > self.rolling:
                            context[0].pop(0)
                            context[1].pop(0)
                    else:
                        context.append(int(target[r]))
                        if len(context) > self.rolling:
                            context.pop(0)
                else:
                    context0 = int(target[r])
                    context1 = 1
                    if self.weighted:
                        context0 *= weights[r]
                        context1 *= weights[r]
                    if self.decay:
                        key = tuple([v[r] for v in values])
                        if key not in decayed:
                            decayed.add(key)  # We don't want to decay twice the same data
                            context[0] = context[0] * self.decay
                            context[1] = context[1] * self.decay
                    context[0] += context0
                    context[1] += context1
                    if self.time_since_last:
                        context[2] = int(timestamp[r])
                        context[3] = int(task_container_id[r])

            except KeyError:
                key = tuple([v[r] for v in values])
                decayed.add(key)
                context0 = int(target[r])
                context1 = 1
                if self.weighted:
                    context0 *= weights[r]
                    context1 *= weights[r]
                context = self._add_to_context([v[r] for v in values], [int(target[r])], [weights[r]], [context0, context1])
                if self.time_since_last:
                    context.append(int(timestamp[r]))
                    context.append(task_container_id[r])

    def update_transform(self, X, y=None):
        self.update(X, y)
        return X

    def get_user_context(self, user_id):
        return self.context[user_id]

    def set_user_context(self, user_id, context):
        self.context[user_id] = context


class RatioEncoder:

    def __init__(self, column, updatable=False):
        self.column = column
        self.updatable = updatable
        self.ratio_name = '{}_ratio'.format(column)
        self.nrows = None
        self.ratios = {}

    def fit(self, X, y=None):
        logging.info('- Fit ratio encoder for {}'.format(self.column))
        self.nrows = len(X)
        self.ratios = X.groupby(self.column).size().to_dict()
        return self

    def transform(self, X):
        if self.updatable:
            self._update_pre_transform(X)
        X = self._merge_context(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X = self._merge_context(X)
        return X

    def _merge_context(self, X):
        values = X[self.column].values
        rows = len(values)
        results = np.zeros(rows, dtype=FLOAT_DTYPE)
        for r in range(rows):
            try:
                results[r] = self.ratios[values[r]] / self.nrows
            except:
                results[r] = 0  # Should never happen with full data

        X[self.ratio_name] = results
        return X

    def _update_pre_transform(self, X):
        self.nrows += len(X)
        values = X[self.column].values
        for v in values:
            self.ratios[v] += 1

    def update_transform(self, X, y=None):
        return X
