import logging
import numpy as np

from doppel import terminate
from doppel.aws.s3 import S3Bucket

from riiid.config import PARAMS
from riiid.core.neural import NeuralModel
from riiid.aws.config import CONTEXT


CONTEXT.get_logger()

try:
    logging.info('Loading data')
    bucket = S3Bucket('model-20201219-093629')
    X = bucket.load_pickle('X.pkl')
    y = bucket.load_pickle('y.pkl')
    train = bucket.load_pickle('train.pkl')
    valid = bucket.load_pickle('valid.pkl')

    nn = NeuralModel(PARAMS['mlp_params'])
    nn.fit(X[train], y[train], X[valid], y[valid])

    bucket.save_pickle_multiparts(nn.save())

except Exception as e:
    logging.info('Unexpected exception: ' + str(e))

finally:
    terminate(CONTEXT)
