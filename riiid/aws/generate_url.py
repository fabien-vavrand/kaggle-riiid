from doppel.aws.s3 import S3Client

client = S3Client()
url = client.create_presigned_url("doppel-riiid-train", "model_20201208_095232.zip")
print(url)
