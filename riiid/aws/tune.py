import os
import uuid
import logging

from doppel import terminate

from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.optim import draw_params
from riiid.config import PARAMS
from riiid.aws.config import CONTEXT


CONTEXT.get_logger()


def score_params(params, n_users=30000):
    loader = DataLoader(CONTEXT.data_path())
    train, questions, lectures = loader.load_first_users(n_users)
    questions = preprocess_questions(questions)
    lectures = preprocess_lectures(lectures)

    test = loader.load_tests('tests_0.pkl')
    train = merge_test(train, test)
    del test

    model = RiiidModel(questions, lectures, params)
    X, y, train, valid = model.fit_transform(train)
    model.fit_lgbm(X[train], y[train], X[valid], y[valid])

    return model.best_score, model.best_iteration


try:
    while True:
        params = draw_params()
        params = PARAMS.update(params)
        best_score, best_iter = score_params(params)
        results = {
            'params': params,
            'best_score': best_score,
            'best_iteration': best_iter
        }
        CONTEXT.save_json(results, os.path.join('results', str(uuid.uuid4()) + '.json'))

except Exception as e:
    logging.info(str(e))

finally:
    terminate(CONTEXT)
