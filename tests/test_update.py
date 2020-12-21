import os
import logging
import numpy as np
import pandas as pd
from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures, save_pkl, load_pkl
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH, PARAMS, TEST_PATH


def get_data(n=1000):
    loader = DataLoader(INPUT_PATH)
    train, questions, lectures = loader.load_first_users(n)
    questions = preprocess_questions(questions)
    lectures = preprocess_lectures(lectures)

    test = loader.load_tests('tests_1.pkl')
    return train, questions, lectures, test


def generate_reference_and_validation_datasets(n=1000, validation_ratio=0.5):
    # Reference data
    train, questions, lectures, test = get_data(n)
    train_reference = merge_test(train, test)
    model = RiiidModel(questions, lectures, PARAMS)
    X_reference, *_ = model.fit_transform(train_reference)
    model.save(os.path.join(TEST_PATH, 'model_ref.zip'))

    # Compare data
    train, questions, lectures, test = get_data(n)
    train_compare, validation = merge_test(train, test, validation_ratio=validation_ratio)
    model = RiiidModel(questions, lectures, PARAMS)
    X_compare, y, train, valid = model.fit_transform(train_compare)
    model.fit_model(X_compare[train], y[train], X_compare[valid], y[valid])

    # Loading model
    model.save(os.path.join(TEST_PATH, 'model_test.zip'))
    model: RiiidModel = RiiidModel.load(os.path.join(TEST_PATH, 'model_test.zip'))

    X_validation = []
    for test in validation:
        test = model.update(test)
        X, predictions = model.predict(test)
        if len(X) > 0:
            X_validation.append(X)

    validation = pd.concat(validation)
    X_validation = pd.concat(X_validation)

    data = (train_reference, X_reference, validation, X_validation)
    return data


def remove_lectures(x):
    return x[x['content_type_id'] == 0].drop(columns=['content_type_id'])


def add_infos(X, train):
    X['user_id'] = train['user_id'].values
    X['content_id'] = train['content_id'].values
    X['task_container_id'] = train['task_container_id'].values
    X = X.sort_values(['user_id', 'timestamp'])
    return X


def build_ref_and_val_datasets(train_reference, X_reference, validation, X_validation):
    train_reference = remove_lectures(train_reference)
    validation = remove_lectures(validation)

    X_reference = add_infos(X_reference, train_reference)
    X_validation = add_infos(X_validation, validation)

    keys = X_validation[['user_id', 'task_container_id']].drop_duplicates()
    X_ref = pd.merge(keys, X_reference, on=['user_id', 'task_container_id'], how='left')
    X_ref = X_ref.reset_index(drop=True)
    X_validation = X_validation.reset_index(drop=True)
    X_validation = X_validation[X_ref.columns]

    return X_ref, X_validation


def is_different(a, b):
    if np.isnan(a) and np.isnan(b):
        return False
    if np.isnan(a) or np.isnan(b):
        return True
    elif a != b:
        return True


def is_nan_different(a, b):
    if np.isnan(a) or np.isnan(b):
        return True
    else:
        return False


def compute_column_differences(X_ref, X_val, column):
    n_equals = 0
    nan_diff = []
    diff = []
    for (i, row_ref), (j, row_val) in zip(X_ref.iterrows(), X_val.iterrows()):
        a, b = row_ref[column], row_val[column]
        if is_different(a, b):
            if is_nan_different(a, b):
                nan_diff.append([i, a, b])
            else:
                diff.append([i, a, b])
        else:
            n_equals += 1
    return diff, n_equals, nan_diff


def compute_differences(X_ref, X_val):
    logging.info('Computing columns differences')
    results = {}
    nan_differences = {}
    differences = {}

    for column in X_ref.columns:
        logging.info(column)
        diff, n_equals, nan_diff = compute_column_differences(X_ref, X_val, column)
        analysis = {
            'type_ref': str(X_ref[column].dtype),
            'type_val': str(X_val[column].dtype),
            'equals': n_equals,
            'nan_differents': len(nan_diff),
            'differents': len(diff)
        }
        if len(diff) > 0:
            diffs = [b - a for _, a, b in diff]
            analysis['min_difference'] = np.min(diffs)
            analysis['q01_difference'] = np.quantile(diffs, 0.01)
            analysis['q10_difference'] = np.quantile(diffs, 0.1)
            analysis['q20_difference'] = np.quantile(diffs, 0.2)
            analysis['mean_difference'] = np.mean(diffs)
            analysis['q80_difference'] = np.quantile(diffs, 0.8)
            analysis['q90_difference'] = np.quantile(diffs, 0.9)
            analysis['q99_difference'] = np.quantile(diffs, 0.99)
            analysis['max_difference'] = np.max(diffs)
            differences[column] = diff
        if len(nan_diff) > 0:
            nan_differences[column] = nan_diff
        results[column] = analysis

    results = pd.DataFrame(results).transpose().reset_index()
    return results, nan_differences, differences


configure_console_logging()

data = generate_reference_and_validation_datasets(n=10000, validation_ratio=0.1)
save_pkl(data, path=os.path.join(TEST_PATH, 'test_ref_val.pkl'))
data = load_pkl(os.path.join(TEST_PATH, 'test_ref_val.pkl'))
X_ref, X_val = build_ref_and_val_datasets(*data)
results, nan_differences, differences = compute_differences(X_ref, X_val)

save_pkl((X_ref, X_val, results, nan_differences, differences), path=os.path.join(TEST_PATH, 'test_results.pkl'))
