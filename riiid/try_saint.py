import os
import time
import logging

from riiid.core.data import DataLoader, save_pkl, load_pkl
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH


configure_console_logging()

model_id = 'saint_20210101_134548'
model: SaintModel = SaintModel.load(MODELS_PATH, model_id)

"""
import tensorflow as tf
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    # model.model = tf.saved_model.load(os.path.join(MODELS_PATH, 'model'))
"""
model.load_model_from_path(os.path.join(MODELS_PATH, 'model'))

loader = DataLoader(INPUT_PATH)
tests = loader.load_tests_examples()

for i, test in enumerate(tests):
    if model.test_batch == 1:
        start = time.perf_counter()

    test = model.update(test)
    _, predictions = model.predict(test)
    print(predictions)

end = time.perf_counter()
total = end - start
logging.info('Time spent: {:.1f}s ({:.3f}s by batch)'.format(total, total / (model.test_batch-1)))
