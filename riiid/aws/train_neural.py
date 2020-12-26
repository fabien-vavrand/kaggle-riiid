import time
import logging
import numpy as np

from doppel import DoppelProject, DoppelContext
from doppel.aws.s3 import S3Bucket

from riiid.config import PARAMS
from riiid.core.neural import NeuralModel

context = DoppelContext()
context.get_logger()

try:
    logging.info("Loading data")
    bucket = S3Bucket("model-20201219-093629")
    X = bucket.load_pickle("X")
    X = X.to_numpy(dtype=np.float32)
    y = bucket.load_pickle("y")
    train = bucket.load_pickle("train")
    valid = bucket.load_pickle("valid")

    nn = NeuralModel(PARAMS["mlp_params"])
    nn.fit(X[train], y[train], X[valid], y[valid])

    bucket.save_pickle_multiparts(nn.save())

except Exception as e:
    logging.info("Unexpected exception: " + str(e))

finally:
    logging.info("Finished")
    time.sleep(30)
    if context.is_doppel:
        DoppelProject(context.doppel_name).terminate()
