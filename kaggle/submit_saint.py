import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


import sys
import logging
PATH = '/kaggle/input/riiid-saint-model'
sys.path.append(PATH)


from riiid.utils import configure_console_logging, check_versions
from riiid.saint.model import SaintModel


configure_console_logging()
check_versions()

logging.info('Load model')
MODEL_ID = 'saint_20210101_132425'
model: SaintModel = SaintModel.load(PATH, MODEL_ID)
model.load_model_from_path('gs://riiid-models/{}_model'.format(MODEL_ID))

for test, _ in iter_test:
    test = model.update(test)
    _, predictions = model.predict(test)
    env.predict(predictions)
