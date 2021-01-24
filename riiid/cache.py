import os

from riiid.config import PATH, ACTIVATE_CACHE
from riiid.core.data import load_pkl, save_pkl


class CacheManager:

    def __init__(self, path, activated=True):
        self.path = os.path.join(path, '.cache')
        self.activated = activated

    def _get_path(self, cache_id):
        return os.path.join(self.path, cache_id + '.pkl')

    def exists(self, cache_id):
        if not self.activated:
            return False
        if os.path.exists(self._get_path(cache_id)):
            return True
        else:
            return False

    def save(self, data, cache_id):
        if self.activated:
            save_pkl(data, self._get_path(cache_id))

    def load(self, cache_id):
        return load_pkl(self._get_path(cache_id))


def get_cache_manager():
    return CACHE_MANAGER


CACHE_MANAGER = CacheManager(PATH, ACTIVATE_CACHE)
