import os
import json
from doppel.aws.s3 import S3Client, S3Bucket


bucket = S3Bucket("doppel-riiid-tune")
_, files = bucket.listdir("results")
for file in files:
    data = bucket.load_json(os.path.join("results", file))
    with open(os.path.join(r"C:\Users\chass\Kaggle\riiid\tuning\params", file), "w") as file:
        json.dump(data, file)
