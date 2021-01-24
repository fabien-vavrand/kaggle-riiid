import logging

from doppel import terminate
from doppel.aws.s3 import S3Bucket

from riiid.core.data import DataLoader
from riiid.saint.model import SaintModel
from riiid.aws.config import CONTEXT


CONTEXT.get_logger()

try:
    loader = DataLoader(CONTEXT.data_path())
    train, questions, lectures = loader.load()

    model = SaintModel(questions, lectures)
    train = model.fit_transform(train)

    train, test = model.split_train_test(train)
    train = model.create_features(train)
    test = model.create_features(test)
    X_train, y_train = model.create_dataset(train)
    X_test, y_test = model.create_dataset(test)

    bucket = S3Bucket(model.get_normalized_name())
    logging.info('Saving model')
    bucket.save_pickle(model, model.get_name(ext='pkl'))

    logging.info('Saving data')
    bucket.save_pickle_multiparts((X_train, y_train, X_test, y_test), model.get_name('data.pkl'))

except Exception as e:
    logging.info(str(e))

finally:
    terminate(CONTEXT)
