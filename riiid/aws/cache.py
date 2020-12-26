from doppel.aws.s3 import S3Bucket


class S3CacheManager:
    def __init__(self, bucket, activated=True):
        self.bucket = S3Bucket(bucket)
        self.activated = activated

    def _get_path(self, cache_id):
        return cache_id + ".pkl"

    def exists(self, cache_id):
        if not self.activated:
            return False
        if self.bucket.exists(self._get_path(cache_id)):
            return True
        else:
            return False

    def save(self, data, cache_id):
        if self.activated:
            self.bucket.save_pickle(data, self._get_path(cache_id))

    def load(self, cache_id):
        return self.bucket.load_pickle(self._get_path(cache_id))
