import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


import os
import sys
import logging
PATH = '/kaggle/input/riiid-submission'
sys.path.append(PATH)


from riiid.utils import configure_console_logging, check_versions
from riiid.core.model import RiiidModel


configure_console_logging()
check_versions()

logging.info('Load model')
model = RiiidModel.load(os.path.join(PATH, 'model'))

for test, _ in iter_test:
    test = model.update(test)
    _, predictions = model.predict(test)
    env.predict(predictions)
