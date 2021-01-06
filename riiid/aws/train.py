import time
import logging

from doppel import DoppelProject
from doppel.aws.s3 import S3Bucket

from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.config import PARAMS
from riiid import cache

from riiid.aws.cache import S3CacheManager
from riiid.aws.train_start import CONTEXT


CONTEXT.get_logger()


try:
    cache.CACHE_MANAGER = S3CacheManager('kaggle-riiid-cache')

    loader = DataLoader(CONTEXT.data_path())
    train, questions, lectures = loader.load()
    questions = preprocess_questions(questions)
    lectures = preprocess_lectures(lectures)

    test = loader.load_tests('tests_0.pkl')
    train = merge_test(train, test)
    del test

    PARAMS['question_embedding']['workers'] = 32
    PARAMS['answers_embedding']['workers'] = 32
    model = RiiidModel(questions, lectures, params=PARAMS)
    X, y, train, valid = model.fit_transform(train)

    logging.info('Saving unfitted model')
    bucket = S3Bucket(model.get_normalized_name())
    bucket.save_multiparts(model.save_with_source(), model.get_name())

    logging.info('Saving data')
    for data, name in [(X, 'X.pkl'), (y, 'y.pkl'), (train, 'train.pkl'), (valid, 'valid.pkl')]:
        bucket.save_pickle_multiparts(data, name)

    model.fit_lgbm(X[train], y[train], X[valid], y[valid])
    bucket.save_pickle_multiparts(model.models[-1], model.get_name('lgbm.pkl'))

    model.fit_catboost(X[train], y[train], X[valid], y[valid])
    bucket.save_pickle_multiparts(model.models[-1], model.get_name('catboost.pkl'))

except Exception as e:
    logging.info(str(e))

finally:
    logging.info('Finished')
    time.sleep(30)
    if CONTEXT.is_doppel:
        DoppelProject(CONTEXT.doppel_name).terminate()
