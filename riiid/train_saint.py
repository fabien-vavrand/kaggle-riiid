import os
import numpy as np
import pandas as pd
from riiid.core.data import DataLoader, save_pkl
from riiid.core.utils import DataFrameAnalyzer
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH


configure_console_logging()

loader = DataLoader(INPUT_PATH)
train, questions, lectures = loader.load_first_users(1000)

model = SaintModel(questions, lectures)
train = model.fit_transform(train)

DataFrameAnalyzer().summary(train)

train, test = model.split_train_test(train)
train = model.create_features(train)
test = model.create_features(test)
X_train, y_train = model.create_dataset(train)
X_test, y_test = model.create_dataset(test)

save_pkl((X_train, y_train, X_test, y_test), os.path.join(MODELS_PATH, model.get_name('data.pkl')))

model.train(X_train, y_train, X_test, y_test)
for keep in [None, 20, 10, 5, 2, 1]:
    model.score(X_test, y_test, keep_last=keep)

# Saving model
model.save(MODELS_PATH)
