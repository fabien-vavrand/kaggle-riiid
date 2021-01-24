import logging

from doppel import terminate
from doppel.aws.s3 import S3Bucket

from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.config import PARAMS
from riiid import cache
from riiid.aws.cache import S3CacheManager
from riiid.aws.config import CONTEXT


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

    bucket = S3Bucket(model.get_normalized_name())

    logging.info('Saving data')
    for data, name in [(X, 'X'), (y, 'y'), (train, 'train'), (valid, 'valid')]:
        bucket.save_pickle_multiparts(data, name + '.pkl')

    model.fit_lgbm(X[train], y[train], X[valid], y[valid])
    model.fit_catboost(X[train], y[train], X[valid], y[valid])

    logging.info('Saving model')
    bucket.save_multiparts(model.save_with_source(), model.get_name())

except Exception as e:
    logging.info('Unexpected exception: ' + str(e))

finally:
    terminate(CONTEXT)
