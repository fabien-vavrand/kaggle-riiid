import logging
import numpy as np

try:
    from numba import njit
    from numba.core import types
    from numba.typed import Dict
except:
    logging.warning('Numba not properly loaded')


@njit
def rolling_count(data):
    rows = data.shape[0]
    cols = data.shape[1]
    results = np.zeros((rows, 1), dtype=np.int16)
    col_ids = [-1 for _ in range(cols)]
    for i in range(rows):
        for c in range(cols):
            if col_ids[c] != data[i, c]:
                col_ids[c] = data[i, c]
                sum = 0
        sum += 1
        results[i, 0] = sum
    return results


@njit
def rolling_sum(data, dtype=np.int16):
    # By default, we sum the last column
    rows = data.shape[0]
    cols = data.shape[1] - 1
    results = np.zeros((rows, 1), dtype=dtype)
    col_ids = [-1 for _ in range(cols)]
    for i in range(rows):
        for c in range(cols):
            if col_ids[c] != data[i, c]:
                col_ids[c] = data[i, c]
                sum = 0
        sum += data[i, cols]
        results[i, 0] = sum
    return results


def last_lecture(X, column):
    return last_lecture_feature(X['user_id'].values, X['content_type_id'].values, X[column].values)


@njit
def last_lecture_feature(users, content_types, feature):
    # By default, we sum the last column
    rows = users.shape[0]
    results = np.zeros((rows, 1), dtype=np.float64)
    user_id = -1
    for r in range(rows):
        if users[r] != user_id:
            user_id = users[r]
            last_feature = np.nan
        if content_types[r] == 1:
            last_feature = feature[r]
        results[r] = last_feature
    return results


@njit
def rolling_categories_count(data, n_categories):
    # Last column is the category. nan are expected to be -1
    rows = data.shape[0]
    cols = data.shape[1] - 1
    results = np.zeros((rows, n_categories), dtype=np.int16)
    col_ids = np.array([-1 for _ in range(cols)])
    for i in range(rows):
        for c in range(cols):
            if col_ids[c] != data[i, c]:
                col_ids[c] = data[i, c]
                sum = np.zeros(n_categories)
        if not np.isnan(data[i, c]):
            sum[int(data[i, cols])] += 1
        results[i, :] = sum
    return results


def rolling_score(data, data_weights, rolling, weighted, decay, dtype):
    # Expecting columns + timestamp + task + target

    cols = data.shape[1]
    sort_cols = cols - 2  # We sort on columns + timestamp
    sort_ids = np.lexsort([data[:, i] for i in reversed(range(sort_cols))])
    data = data[sort_ids, :]
    data_weights = data_weights[sort_ids]

    results, keys, values, weights, context = sorted_rolling_score(data, data_weights, rolling, weighted, decay, dtype)

    reverse_sort_ids = np.lexsort([sort_ids])
    results = results[reverse_sort_ids, :]
    return results, keys, values, weights, context


@njit
def sorted_rolling_score(data, data_weights, rolling, weighted, decay, dtype):
    # Expecting columns + timestamp + task + target
    # sum_pop and count_pop are used to avoid poping element when rolling in the middle of a task

    rows = data.shape[0]
    cols = data.shape[1]
    target_col = cols - 1
    task_col = cols - 2
    group_cols = cols - 3  # We group on columns
    results = np.zeros((rows, 2), dtype=dtype)
    col_ids = np.array([-1 for _ in range(group_cols)])
    task_id = -1
    keys = []
    values = []
    weights = []
    context = []  # contains the last values for each group. Required because of decay which can't be recomputed on rolling values
    sum = 0
    count = 0

    for r in range(rows):
        same_group = True
        for c in range(group_cols):
            if col_ids[c] != data[r, c]:
                col_ids[c] = data[r, c]
                same_group = False
        if not same_group:
            # Only add context after compute on the first group
            if len(keys) > 0:
                context.append([sum, count])
            sum = 0
            count = 0
            roll = []
            roll_weights = []
            sum_task = 0
            count_task = 0
            sum_pop = 0
            count_pop = 0
            keys.append(col_ids.copy())
            values.append(roll)
            weights.append(roll_weights)
        if task_id != data[r, task_col]:
            task_id = data[r, task_col]
            sum_task = 0
            count_task = 0
            sum -= sum_pop
            count -= count_pop
            sum_pop = 0
            count_pop = 0
            if decay is not None:
                sum = sum * decay
                count = count * decay
        results[r, 0] = sum - sum_task
        results[r, 1] = count - count_task
        if weighted:
            sum += data[r, target_col] * data_weights[r]
            sum_task += data[r, target_col] * data_weights[r]
            count += data_weights[r]
            count_task += data_weights[r]
        else:
            sum += data[r, target_col]
            sum_task += data[r, target_col]
            count += 1
            count_task += 1
        roll.append(data[r, target_col])
        roll_weights.append(data_weights[r])
        if rolling is not None and count > rolling:
            # We consider the task is shorter than the rolling period, else we sould also decrement sum_task and count_task
            target = roll.pop(0)
            weight = roll_weights.pop(0)
            if weighted:
                sum_pop += target * weight
                count_pop += weight
            else:
                sum_pop += target
                count_pop += 1
    context.append([sum, count])
    return results, keys, values, weights, context


def last_feature_value_time(X, time_feature, feature_name):
    user_id = X['user_id'].values
    task_container_id = X['task_container_id'].values
    timestamp = X[time_feature].values
    feature = X[feature_name].values
    return _last_feature_value_time(user_id, task_container_id, timestamp, feature)


@njit
def _last_feature_value_time(user_id, task_container_id, timestamp, feature):
    results = np.full(len(user_id), np.nan, dtype=np.float64)
    uid = -1
    task_id = -1

    for r in range(len(user_id)):
        if uid != user_id[r]:
            uid = user_id[r]
            user_context = Dict.empty(
                key_type=types.int64,
                value_type=types.int64
            )
            task_context = Dict.empty(
                key_type=types.int64,
                value_type=types.int64
            )
        if task_id != task_container_id[r]:
            task_id = task_container_id[r]
            for f, ts in task_context.items():
                user_context[f] = ts
            task_context = Dict.empty(
                key_type=types.int64,
                value_type=types.int64
            )
        f = feature[r]
        if f in user_context:
            results[r] = user_context[f]
        task_context[f] = timestamp[r]
    return results


def compute_user_answers_ratio(X, decay):
    return _compute_user_answers_ratio(X['user_id'].values, X['user_answer'].values, X['correct_answer'].values, X[[0, 1, 2, 3]].values, decay)


@njit
def _compute_user_answers_ratio(user_id, user_answer, correct_answer, answers_ratio, decay):
    results = np.zeros((len(user_id), 2))
    users = []
    contexts = []
    user_context = np.zeros(5)
    uid = - 1
    for r in range(len(user_id)):
        if user_id[r] != uid:
            if uid != -1:
                users.append(uid)
                contexts.append(user_context.copy())
            uid = user_id[r]
            answers = np.zeros(4)
            questions = np.zeros(4)
            user_context = np.zeros(8)
        results[r, 0] = answers[correct_answer[r]]
        results[r, 1] = questions[correct_answer[r]]
        answers = decay * answers
        answers[user_answer[r]] += 1
        questions = decay * questions + answers_ratio[r, :]
        user_context[:4] = answers
        user_context[4:] = questions
    users.append(uid)
    contexts.append(user_context.copy())
    return results, users, contexts
