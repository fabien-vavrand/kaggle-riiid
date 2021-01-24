import os
from riiid.config import INPUT_PATH, APPS_PATH
from doppel import DoppelContext


CONTEXT = DoppelContext() \
    .add_data(key='train.pkl', bucket='kaggle-riiid', source=os.path.join(INPUT_PATH, 'train.pkl')) \
    .add_data(key='tests_0.pkl', bucket='kaggle-riiid', source=os.path.join(INPUT_PATH, 'tests_0.pkl')) \
    .add_data(key='tests_1.pkl', bucket='kaggle-riiid', source=os.path.join(INPUT_PATH, 'tests_1.pkl'))


PACKAGES = [
    os.path.join(APPS_PATH, 'aws-doppel')
]
