from doppel import DoppelProject
from riiid.utils import configure_console_logging
from riiid.config import SRC_PATH
from riiid.aws.config import PACKAGES


configure_console_logging()

project = DoppelProject(
    name='riiid-train-neural',
    path=SRC_PATH,
    entry_point='-m riiid.aws.train_neural',
    packages=PACKAGES,
    python='3.7.6',
    n_instances=1,
    min_memory=128,
    min_gpu=1,
    env_vars={'PYTHONHASHSEED': '1'}
)

project.start()
