import math
import logging
import numpy as np
import pandas as pd


def update_pipeline(pipeline, X, y=None):
    for name, transformer in pipeline.steps:
        X = update_transformer(transformer, X, y)
    return X


def update_transformer(transformer, X, y=None):
    if hasattr(transformer, 'update_transform'):
        return transformer.update_transform(X, y)
    else:
        if hasattr(transformer, 'update'):
            transformer.update(X, y)
        return transformer.transform(X)


class DataFrameAnalyzer:

    def __init__(self, width=120, minimal=True, nulls_percent_after=0.01, uniques_percent_after=0.01):
        self.width = width
        self.minimal = minimal

    def _build(self, df):
        self.rows, self.columns = df.shape
        self.space_to_display_number_of_rows = len(str(self.rows))
        self.column_name_width = max([len(str(c)) for c in df.columns])
        self.type_width = max([len(str(df[c].dtype)) for c in df.columns])

        self.header = '{}column | {}nulls | {}uniques | {}type | description{}'.format(
            ' ' * max(self.column_name_width - 6, 0),
            ' ' * max(self.space_to_display_number_of_rows - 5, 0),
            ' ' * max(self.space_to_display_number_of_rows - 7, 0),
            ' ' * max(self.type_width - 4, 0),
            ' ' * self.width
        )[:self.width]

        self.header_separator = '+'.join(['-' * len(c) for c in self.header.split('|')])

        self.row_format = '{{:>{}}} | {{:>{}}} | {{:>{}}} | {{:>{}}} | {{:}}'.format(
            max(self.column_name_width, 6),
            max(self.space_to_display_number_of_rows, 5),
            max(self.space_to_display_number_of_rows, 7),
            max(self.type_width, 4),
        )

        self.description_width = len(self.header.split('|')[-1]) - 1

    def summary(self, df):
        for info in self.analyze(df):
            logging.info(info)

    def analyze(self, df):
        self._build(df)
        yield '{:,} rows x {:,} columns'.format(self.rows, self.columns)
        yield self.header
        yield self.header_separator
        for column in df.columns:
            column_name = str(column)
            n_nulls = df[column].isnull().sum()
            n_unique = self._get_nunique(df[column])
            description = self._get_description(df[column])
            row = self.row_format.format(column_name, n_nulls, n_unique, str(df[column].dtype), description)
            yield row[:self.width]

    def _get_nunique(self, series):
        try:
            return series.nunique()
        except:
            return ''

    def _get_description(self, series):
        if pd.api.types.is_bool_dtype(series):
            return self._get_category_description(series)

        elif pd.api.types.is_float_dtype(series):
            return self._get_float_description(series)

        elif pd.api.types.is_datetime64_any_dtype(series):
            return self._get_date_description(series)

        elif pd.api.types.is_integer_dtype(series):
            return self._get_category_description(series)

        elif pd.api.types.is_categorical_dtype(series):
            return self._get_category_description(series)

        else:
            return self._get_object_description(series)

    def _get_float_description(self, series):
        # chars = [' ', '\u2581', '\u2582', '\u2583', '\u2585', '\u2586', '\u2587']
        smin = series.min()
        smax = series.max()
        smed = series.median()
        description = '[min={:.1f}, median={:.1f}, max={:.1f}]'.format(smin, smed, smax)
        if len(description) > self.description_width:
            description = '[median={:.1f}]'.format(smed)
        if len(description) > self.description_width:
            description = ''
        return description

    def _get_date_description(self, series):
        dmin = series.min().strftime('%Y-%m-%d')
        dmax = series.max().strftime('%Y-%m-%d')
        description = '[{} to {}]'.format(dmin, dmax)
        if len(description) > self.description_width:
            description = ''
        return description

    def _get_category_description(self, series):
        description = ''
        values = series.value_counts()
        for i, v in values.items():
            value_description = '{} ({:.0%})'.format(i, v / self.rows)
            if description == '':
                if len(value_description) <= self.description_width:
                    description = value_description
                else:
                    break
            else:
                if len(description) + len(value_description) <= self.description_width - 2:
                    description += ', ' + value_description
                else:
                    break
        return description

    def _get_object_description(self, series):
        types = series[-pd.isnull(series)].apply(lambda x: type(x))
        type_counts = types.value_counts()
        if len(type_counts) == 0:
            return '<empty>'
        elif len(type_counts) > 1:
            return self._get_category_description(types)
        elif types.values[0] == str:
            return self._get_category_description(series)
        else:
            return str(types.values[0])


def indexed_merge(x, y, left_on):
    return pd.merge(x, y, left_on=left_on, right_index=True, how='left')


def pre_filtered_indexed_merge(x, y, left_on):
    return pd.merge(x, y.loc[y.index.isin(x[left_on])], left_on=left_on, right_index=True, how='left')


def fast_merge(x, y, left_on):
    return pd.concat([x.reset_index(drop=True), y.reindex(x[left_on].values).reset_index(drop=True)], axis=1)


def tasks_bucket_12(r):
    if r <= -2:
        return -2
    elif r <= 2:
        return r
    if r <= 10:
        return 10
    elif r <= 20:
        return 20
    elif r <= 30:
        return 30
    elif r <= 50:
        return 50
    elif r <= 100:
        return 100
    elif r <= 300:
        return 300
    else:
        return 500


def tasks_bucket_3(r):
    if r <= 0:
        return 0
    elif r <= 2:
        return 1
    else:
        return 3


def tasks_group(r):
    if r <= 15:
        return 15
    elif r < 30:
        return 30
    elif r == 30:
        return 40
    elif r <= 50:
        return math.ceil(r/10) * 10
    elif r <= 1000:
        return math.ceil(r / 100) * 100
    else:
        return math.ceil(r / 1000) * 1000
