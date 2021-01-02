import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


import sys
sys.path.append('/kaggle/input/riiidsource')


import logging
from riiid.utils import configure_console_logging, check_versions
from riiid.saint.model import SaintModel


configure_console_logging()
# check_versions()

logging.info('Load model')
path = '/kaggle/input/riiid-saint-model-v0'
model_id = 'saint_20210101_132425'
model: SaintModel = SaintModel.load(path, model_id)
model.load_model_from_path('gs://riiid-models/saint_20201229_122345_model')

for test, _ in iter_test:
    test = model.update(test)
    _, predictions = model.predict(test)
    env.predict(predictions)


