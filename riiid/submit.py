import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()


import sys

sys.path.append("/kaggle/input/riiidsubmissions")


import logging
from riiid.utils import configure_console_logging, check_versions
from riiid.core.model import RiiidModel


configure_console_logging()
check_versions()

logging.info("Load model")
path = "/kaggle/input/riiidsubmissions/model"
model = RiiidModel.load(path)

for test, _ in iter_test:
    test = model.update(test)
    _, predictions = model.predict(test)
    env.predict(predictions)
