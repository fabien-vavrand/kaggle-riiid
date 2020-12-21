import time
import logging
import datetime
from doppel import DoppelProject
from riiid.aws.train_start import CONTEXT


CONTEXT.get_logger()

try:

    while True:
        logging.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(60)

except Exception as e:
    logging.info(str(e))
    time.sleep(30)
    if CONTEXT.is_doppel:
        DoppelProject(CONTEXT.doppel_name).terminate()
