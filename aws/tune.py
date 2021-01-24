from doppel import DoppelProject
from riiid.utils import configure_console_logging
from riiid.config import SRC_PATH
from riiid.aws.config import PACKAGES, CONTEXT


configure_console_logging()

project = DoppelProject(
    name='riiid-tune',
    path=SRC_PATH,
    entry_point='-m riiid.aws.tune',
    packages=PACKAGES,
    python='3.7.6',
    n_instances=10,
    min_memory=16,
    context=CONTEXT
)

project.start()
