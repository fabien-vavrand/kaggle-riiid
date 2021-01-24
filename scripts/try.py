import os
import time
import logging

from riiid.core.data import DataLoader
from riiid.utils import configure_console_logging
from riiid.config import MODELS_PATH, INPUT_PATH
from riiid.core.model import RiiidModel


configure_console_logging()

logging.info('Loading model')
MODEL_NAME = 'model_20210123_210542.zip'
model: RiiidModel = RiiidModel.load(os.path.join(MODELS_PATH, MODEL_NAME))

tests = DataLoader(INPUT_PATH).load_tests_examples()

for i, test in enumerate(tests):
    if model.test_batch == 1:
        start = time.perf_counter()

    test = model.update(test)
    _, predictions = model.predict(test)

end = time.perf_counter()
total = end - start
logging.info('Time spent: {:.1f}s ({:.3f}s by batch)'.format(total, total / model.test_batch))
