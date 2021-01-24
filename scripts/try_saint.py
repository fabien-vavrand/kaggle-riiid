import time
import logging

from riiid.core.data import DataLoader
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH


configure_console_logging()

logging.info('Loading model')
MODEL_ID = 'saint_20210104_024103'
model: SaintModel = SaintModel.load(MODELS_PATH, MODEL_ID)

tests = DataLoader(INPUT_PATH).load_tests_examples()

for i, test in enumerate(tests):
    if model.test_batch == 1:
        start = time.perf_counter()

    test = model.update(test)
    _, predictions = model.predict(test)

end = time.perf_counter()
total = end - start
logging.info('Time spent: {:.1f}s ({:.3f}s by batch)'.format(total, total / model.test_batch))
