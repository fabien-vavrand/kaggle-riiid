import os
import json
from doppel.aws.s3 import S3Bucket

from riiid.config import TUNE_PATH


bucket = S3Bucket('doppel-riiid-tune')
_, files = bucket.listdir('results')
for file in files:
    data = bucket.load_json(os.path.join('results', file))
    with open(os.path.join(TUNE_PATH, file), 'w') as file:
        json.dump(data, file)
