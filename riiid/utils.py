import numba
import gensim
import logging
import pathlib
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb


def configure_console_logging():
    LOGGING_FORMAT = "%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s"
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def check_versions():
    assert np.__version__ == "1.18.5"
    assert pd.__version__ == "1.1.3"
    assert lgb.__version__ == "2.3.1"
    assert sklearn.__version__ == "0.23.2"
    assert gensim.__version__ == "3.8.3"
    logging.info("Versions: OK")


def downcast_int(series, allow_unsigned=False):
    min_value = series.min()
    max_value = series.max()
    if allow_unsigned:
        types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
    else:
        types = [np.int8, np.int16, np.int32, np.int64]
    for t in types:
        if (
            np.iinfo(t).min <= min_value
            and max_value <= np.iinfo(t).max
            and np.dtype(t).itemsize < series.dtype.itemsize
        ):
            return series.astype(t)
    return series


def make_tuple(ids):
    try:
        if len(ids) == 1:
            return ids[0]
        return tuple(ids)
    except:
        return ids


def make_iterable(ids):
    try:
        return iter(ids)
    except:
        return [ids]


def logging_callback():
    def callback(env):
        if env.iteration % 100 == 0:
            logging.info("[{}] AUC = {:.5%}".format(env.iteration, env.evaluation_result_list[0][2]))

    return callback


def get_tuple_index(X, column):
    return list(X.columns).index(column)


def keys_to_int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x


def get_riiid_directory():
    return str(pathlib.Path(__file__).parent)
