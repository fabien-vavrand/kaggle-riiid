import os
import uuid
import json
import numpy as np
import pandas as pd

from riiid.config import INPUT_PATH, PARAMS, TUNE_PATH
from riiid.core.data import DataLoader, preprocess_lectures, preprocess_questions
from riiid.core.model import RiiidModel
from riiid.validation import merge_test


layers_sizes = [8 + 4 * n for n in range(20)]


def draw_params():
    return {
        "question_embedding": {
            "n_clusters": np.random.randint(30, 100),
            "embedding_size": int(np.random.choice(layers_sizes, 1)[0]),
            "window": np.random.randint(1, 7),
            "min_count": np.random.randint(1, 10),
            "sg": np.random.randint(0, 2),
            "iter": np.random.randint(5, 50),
        },
        "answers_embedding": {
            "n_fold": 5,
            "embedding_size": int(np.random.choice(layers_sizes, 1)[0]),
            "window": np.random.randint(1, 7),
            "min_count": np.random.randint(1, 10),
            "sg": np.random.randint(0, 2),
            "iter": np.random.randint(5, 50),
        },
        "answer_encoder": {
            "smoothing_min": np.random.uniform(0, 8),
            "smoothing_value": np.random.uniform(0.5, 3),
        },
        "score_encoder": {
            "smoothing_min": np.random.uniform(0, 8),
            "smoothing_value": np.random.uniform(0.5, 3),
            "noise": 0.005,
        },
        "user_score_encoder": {
            "smoothing_min": np.random.uniform(0, 8),
            "smoothing_value": np.random.uniform(0.5, 3),
        },
        "user_content_score_encoder": {
            "smoothing_min": np.random.uniform(0, 8),
            "smoothing_value": np.random.uniform(0.5, 3),
        },
        "user_rolling_score_encoder": {
            "smoothing_min": np.random.uniform(0, 8),
            "smoothing_value": np.random.uniform(0.5, 3),
            "rolling": np.random.randint(5, 100),
        },
        "user_weighted_score_encoder": {
            "smoothing_min": np.random.uniform(0, 8),
            "smoothing_value": np.random.uniform(0.5, 3),
        },
    }


def score_params(params, n_users):
    loader = DataLoader(INPUT_PATH)
    train, questions, lectures = loader.load_first_users(n_users)
    questions = preprocess_questions(questions)
    lectures = preprocess_lectures(lectures)

    test = loader.load_tests("tests_0.pkl")
    train = merge_test(train, test)
    del test

    model = RiiidModel(questions, lectures, params)
    X, y, train, valid = model.fit_transform(train)
    model.fit_lgbm(X[train], y[train], X[valid], y[valid])

    return model.best_score, model.best_iteration


def run_mode(mode, n_users):
    params = PARAMS.copy()
    params[mode] = draw_params()[mode]
    score, iters = score_params(params, n_users)
    results = {"mode": mode, "mode_params": params[mode], "params": params, "score": score, "iterations": iters}
    return results


def tune_mode(mode, n_users, path, n=100):
    for _ in range(n):
        results = run_mode(mode, n_users)
        with open(os.path.join(path, str(uuid.uuid4()) + ".json"), "w") as file:
            json.dump(results, file, indent=4)


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def count_modes(path):
    data = [load_json(os.path.join(path, file)) for file in os.listdir(path)]
    data = pd.concat([pd.json_normalize(d) for d in data])
    return data["mode"].value_counts()


def get_data(mode):
    data = [load_json(os.path.join(path, file)) for file in os.listdir(path)]
    data = [pd.json_normalize(d) for d in data if d["mode"] == mode]
    data = pd.concat(data)
    return data


if __name__ == "__main__":
    """
    ('question_embedding', 50),
    ('answer_encoder', 10),
    ('score_encoder', 10),
    ('user_score_encoder', 10),
    ('user_content_score_encoder', 10),
    ('user_rolling_score_encoder', 40),
    ('user_weighted_score_encoder', 10)
    """
    path = os.path.join(TUNE_PATH, "params_questions")
    if not os.path.exists(path):
        os.mkdir(path)
    for mode, n in [("answers_embedding", 100)]:
        print("---" + mode)
        tune_mode(mode, 20000, path, n=n)
