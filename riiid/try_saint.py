import os
import time
import logging

from riiid.core.data import DataLoader, save_pkl, load_pkl
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH


configure_console_logging()

model_id = 'saint_20210104_024103'
model: SaintModel = SaintModel.load(MODELS_PATH, model_id)
#model.load_model_from_path(os.path.join(MODELS_PATH, 'saint_20210101_222633_model_gs'))

loader = DataLoader(INPUT_PATH)
tests = loader.load_tests_examples()

for i, test in enumerate(tests):
    if model.test_batch == 1:
        start = time.perf_counter()

    test = model.update(test)
    _, predictions = model.predict(test)

end = time.perf_counter()
total = end - start
logging.info('Time spent: {:.1f}s ({:.3f}s by batch)'.format(total, total / (model.test_batch-1)))
