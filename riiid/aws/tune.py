import os
import uuid
import time
import logging

from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.optim import draw_params
from riiid.aws.tune_start import CONTEXT

from doppel import DoppelProject


CONTEXT.get_logger()


def score_params(params):
    loader = DataLoader(CONTEXT.data_path())
    train, questions, lectures = loader.load_first_users(30000)
    questions = preprocess_questions(questions)
    lectures = preprocess_lectures(lectures)

    test = loader.load_tests('tests_0.pkl')
    train = merge_test(train, test)
    del test

    model = RiiidModel(questions, lectures, params)
    X, y, train, valid = model.fit_transform(train)
    model.fit_model(X[train], y[train], X[valid], y[valid])

    return model.best_score, model.best_iteration


try:
    while True:
        params = draw_params()
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
    logging.info('Finished')
    time.sleep(30)
    if CONTEXT.is_doppel:
        DoppelProject(CONTEXT.doppel_name).terminate()
