import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop=None, validate=False):
        self.columns_to_drop = columns_to_drop
        self.validate = validate
        self.columns = None

    def fit(self, X, y=None):
        if self.columns_to_drop is not None:
            self.columns = [c for c in X.columns if c not in self.columns_to_drop]
        else:
            self.columns = list(X.columns)
        return self

    def transform(self, X):
        columns = [c for c in self.columns if c in X.columns]
        if self.validate and len(columns) < len(self.columns):
            missing = set(self.columns).difference(columns)
            raise ValueError(f'Missing columns: {missing}')
        return X[columns]

    def update_transform(self, X, y=None):
        return X


class TypesTransformer(TransformerMixin):

    def __init__(self, float_dtype=None):
        self.float_dtype = float_dtype
        self.dtypes = None

    def fit(self, X, y=None):
        if self.float_dtype is not None:
            for column in X.columns:
                if pd.api.types.is_float_dtype(X[column].dtype) and X[column].dtype != self.float_dtype:
                    X[column] = X[column].astype(self.float_dtype)

        self.dtypes = {column: X[column].dtype for column in X.columns}
        return self

    def transform(self, X):
        for column in X.columns:
            if column in self.dtypes and self.dtypes[column] != X[column].dtype:
                X[column] = X[column].astype(self.dtypes[column])
        return X

    def update_transform(self, X, y=None):
        return X


class MemoryUsageLogger(TransformerMixin):

    def __init__(self, name=None, details=False):
        self.name = name
        self.details = details

    def fit(self, X, y):
        logging.info('[{} rows x {} columns] Memory: {:.1f}Mo'.format(
            X.shape[0], X.shape[1], X.memory_usage().sum() / 1024 ** 2)
        )
        if self.details:
            for index, value in X.memory_usage().items():
                if index == 'Index':
                    logging.info('{}{:<40}: {:.1f}Mo'.format(' ' * 17, index, value / 1024 ** 2))
                else:
                    logging.info('[{:<14}] {:<40}: {:.1f}Mo'.format(str(X[index].dtype), index, value / 1024 ** 2))
        return self

    def transform(self, X):
        return X


class RatioTransformer:

    def __init__(self, name, numerator, denominator):
        self.name = name
        self.numerator = numerator
        self.denominator = denominator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.name] = X[self.numerator] / X[self.denominator].replace({0: 0.00001})
        return X

    def update_transform(self, X, y=None):
        return X


class WeightedAnswerTransformer:

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        X = self._transform(X)
        return X

    def update_transform(self, X, y=None):
        X = self._transform(X)
        return X

    def _transform(self, X):
        X['answer_weight'] = (X['answered_correctly'] == 0) * X[self.column] + (X['answered_correctly'] == 1) * (1 - X[self.column])
        #X['answer_weight'] = -((X['answered_correctly'] == 0) * np.log(1-X['content_id_encoded']) + (X['answered_correctly'] == 1) * np.log(X['content_id_encoded']))
        return X


class DensityTransformer:

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['user_id_density'] = (X['user_id_count'] / X['timestamp'] * 1000 * 60 * 60 * 24).fillna(0)
        for column in self.columns:
            X['{}_density'.format(column.replace('_count', ''))] = (X[column] / X['user_id_count']).fillna(0)
        return X
