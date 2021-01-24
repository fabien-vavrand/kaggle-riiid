import os
from riiid.config import INPUT_PATH
from riiid.core.data import DataLoader, save_pkl, load_pkl
from riiid.validation import generate_test


loader = DataLoader(INPUT_PATH)
train, _, _ = loader.load()

test = generate_test(train, size=2_500_000, N=10_000, seed=0)
save_pkl(test, os.path.join(INPUT_PATH, 'tests_0.pkl'))

test = generate_test(train, size=5_000_000, N=20_000, seed=0)
save_pkl(test, os.path.join(INPUT_PATH, 'tests_1.pkl'))
