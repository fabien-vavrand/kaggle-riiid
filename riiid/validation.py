import os
import numpy as np
import pandas as pd
import itertools
import logging

from riiid.core.data import DataLoader, save_pkl, load_pkl
from riiid.config import INPUT_PATH


class StartPicker:
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.length = N - n + 1

    def get_prob(self, i):
        if i < self.n:
            return np.sqrt(self.n / (i + 1))
        elif i > self.length - self.n - 1:
            return np.sqrt(self.n / (self.length - i))
        else:
            return 1

    def get_probs(self):
        p = [self.get_prob(i) for i in range(self.length)]
        psum = np.sum(p)
        p = [prob / psum for prob in p]
        return p

    def random_start(self):
        choices = list(range(self.length))
        p = self.get_probs()
        return np.random.choice(choices, size=1, replace=False, p=p)[0]


def generate_test(X, size, N=10000, seed=0):
    np.random.seed(seed)

    users = X.groupby("user_id")["timestamp"].max().reset_index()
    users.columns = ["user_id", "duration"]
    users["duration"] = users["duration"] / (1000 * 60 * 60 * 24)

    # we pick a random initial timestamp for each user, so that their full period is within the riiid time period
    total_duration = np.ceil(users["duration"].max())
    users["random_period"] = total_duration - users["duration"]
    users["random"] = np.random.random(len(users))
    users["appearance"] = users["random"] * users["random_period"]
    users["appearance"] = users["appearance"] * (1000 * 60 * 60 * 24)
    users["initial_timestamp"] = np.round(users["appearance"], 0).astype(np.int64)

    # We then compute the global timestamp for each task
    X = pd.merge(X, users[["user_id", "initial_timestamp"]], on=["user_id"])
    X["global_timestamp"] = X["initial_timestamp"] + X["timestamp"]

    # We pick the last "size" rows sorted on this global timestamp
    test = X.groupby(["user_id", "task_container_id", "global_timestamp"], sort=False).size()
    test = test.reset_index()
    test.rename(columns={0: "n"}, inplace=True)
    test = test.sort_values("global_timestamp", ascending=False)
    test["cumn"] = test["n"].rolling(len(test), min_periods=1).sum()
    test = test[test["cumn"] <= size]
    test = test.sort_values("global_timestamp")
    test = test.drop(columns=["n", "cumn"])

    test = generate_batches(test, N)
    return test


def generate_batches(test, N):
    # We build more or less equal size batches
    groups = test.groupby("user_id")
    batches = [[] for _ in range(N)]
    for user_id, user_test in groups:
        n = len(user_test)
        i = StartPicker(N, n).random_start()
        for j, row in enumerate(user_test.itertuples(index=False)):
            batches[i + j].append(
                {"batch_id": i + j, "user_id": row.user_id, "task_container_id": row.task_container_id}
            )

    batches = itertools.chain.from_iterable(batches)
    return pd.DataFrame(batches)


def build_test_batches(X):
    # Expected columns for tests:
    COLUMNS = [
        "row_id",
        "timestamp",
        "user_id",
        "content_id",
        "content_type_id",
        "task_container_id",
        "prior_question_elapsed_time",
        "prior_question_had_explanation",
        "prior_group_answers_correct",
        "prior_group_responses",
        "answered_correctly",
    ]

    X = X[-pd.isnull(X["batch_id"])].copy().reset_index(drop=True)
    batches = X.groupby("batch_id")
    batches = [batch.copy() for _, batch in batches]

    for i, batch in enumerate(batches):
        if i == 0:
            prior_user_answer = []
            prior_answered_correctly = []
        else:
            prior_user_answer = list(batches[i - 1]["user_answer"].values)
            prior_answered_correctly = list(batches[i - 1]["answered_correctly"].values)
            batches[i - 1] = batches[i - 1][COLUMNS]
        batch["row_id"] = 0
        batch.reset_index(drop=True, inplace=True)
        batch["prior_group_answers_correct"] = [str(prior_answered_correctly)] + [np.nan] * (len(batch) - 1)
        batch["prior_group_responses"] = [str(prior_user_answer)] + [np.nan] * (len(batch) - 1)

        if i == len(batches) - 1:
            batches[i] = batches[i][COLUMNS]

    return batches


def build_train(X):
    X = X[pd.isnull(X["batch_id"])].copy().reset_index(drop=True)
    X = X.drop(columns=["batch_id"])
    return X


"""
# deprecated
def build_train_test(train, test):
    X = pd.merge(train, test, on=['user_id', 'task_container_id'], how='left')
    test = build_test(X)
    train = build_train(X)

    train_size = len(train)
    validation_size = np.sum([len(b) for b in test])
    validation_ratio = validation_size / (train_size + validation_size)
    logging.info(f'Train size = {train_size}, validation size = {validation_size} [{validation_ratio:.1%}]')

    first_batch_id = np.min(X['batch_id'])
    last_batch_id = np.max(X['batch_id'])
    logging.info(f'{len(test)} batches, from batch id {first_batch_id:.0f}, to batch id = {last_batch_id:.0f}')

    users = set(X['user_id'].values)
    train_users = set(train['user_id'].values)
    validation_users = set(X[-pd.isnull(X['batch_id'])]['user_id'].values)
    known_users = validation_users.intersection(train_users)
    new_users = validation_users.difference(train_users)
    logging.info(f'{len(users)} users, o/w {len(train_users)} in train and {len(validation_users)} in validation ({len(known_users)} existing, {len(new_users)} new)')
    return train, test
"""


def merge_test(train, test, validation_ratio=None, return_batches=True):
    X = pd.merge(train, test, on=["user_id", "task_container_id"], how="left")

    if validation_ratio is not None:
        batches = sorted(test["batch_id"].unique())
        n_validation_batches = int(len(batches) * validation_ratio)
        validation_batches = batches[-n_validation_batches:]
        validation = X[X["batch_id"].isin(validation_batches)]
        X = X[-X["batch_id"].isin(validation_batches)].copy()

    train = X[pd.isnull(X["batch_id"])]
    test = X[-pd.isnull(X["batch_id"])]

    train_size = len(train)
    test_size = len(test)
    test_ratio = test_size / (train_size + test_size)
    logging.info(f"Train size = {train_size}, test size = {test_size} [{test_ratio:.1%}]")

    if validation_ratio is not None:
        validation_size = len(validation)
        logging.info(f"Validation size = {validation_size}")

    users = set(X["user_id"].values)
    train_users = set(train["user_id"].values)
    test_users = set(test["user_id"].values)
    known_users = test_users.intersection(train_users)
    new_users = test_users.difference(train_users)
    logging.info(
        f"{len(users)} users, o/w {len(train_users)} in train and {len(test_users)} in test ({len(known_users)} existing, {len(new_users)} new)"
    )

    if validation_ratio is not None:
        train_test_users = set(X["user_id"].values)
        validation_users = set(validation["user_id"].values)
        known_users = validation_users.intersection(train_test_users)
        new_users = validation_users.difference(train_test_users)
        logging.info(f"{len(validation_users)} users in validation ({len(known_users)} existing, {len(new_users)} new)")

    if validation_ratio is None:
        return X
    else:
        if return_batches:
            validation = build_test_batches(validation)
        return X, validation


if __name__ == "__main__":
    loader = DataLoader(INPUT_PATH)
    train, _, _ = loader.load()

    test = generate_test(train, size=5_000_000, N=20_000, seed=0)
    save_pkl(test, os.path.join(INPUT_PATH, "tests_1.pkl"))
