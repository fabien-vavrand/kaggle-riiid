import os
import time
import logging
import pandas as pd
from riiid.core.data import load_pkl, save_pkl, DataLoader
from riiid.utils import configure_console_logging
from riiid.config import MODELS_PATH, INPUT_PATH
from riiid.core.model import RiiidModel
from riiid.validation import merge_test


configure_console_logging()

logging.info('Loading model')
model: RiiidModel = RiiidModel.load(os.path.join(MODELS_PATH, 'model_20201208_013556.zip'))

loader = DataLoader(INPUT_PATH)
tests = loader.load_tests_examples()

logging.info('Loading tests')
"""
train, _, _ = loader.load_first_users(10000)
test = loader.load_tests('tests_1.pkl')
_, tests = merge_test(train, test, validation_ratio=0.5)
#save_pkl(tests, os.path.join(INPUT_PATH, 'tests_1_batches.pkl'))
"""
#tests = load_pkl(os.path.join(INPUT_PATH, 'tests_1_batches.pkl'))


for i, test in enumerate(tests):
    if model.test_batch == 1:
        start = time.perf_counter()

    if 'answered_correctly' in test.columns:
        test = test.drop(columns='answered_correctly')
    test = model.update(test)
    _, predictions = model.predict(test)

    if model.test_batch == 50:
        break

end = time.perf_counter()
total = end - start
logging.info('Time spent: {:.1f}s ({:.3f}s by batch)'.format(total, total / (model.test_batch-1)))
