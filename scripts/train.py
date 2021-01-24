import os
from riiid.core.data import DataLoader, preprocess_questions, preprocess_lectures, save_pkl
from riiid.core.model import RiiidModel
from riiid.validation import merge_test
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH, PARAMS


configure_console_logging()

# Load and preprocess data
loader = DataLoader(INPUT_PATH)
train, questions, lectures = loader.load_first_users(30000)
questions = preprocess_questions(questions)
lectures = preprocess_lectures(lectures)

# Load and merge validation set
test = loader.load_tests('tests_0.pkl')
train = merge_test(train, test)

# Compute features
model = RiiidModel(questions, lectures, params=PARAMS)
X, y, train, valid = model.fit_transform(train)
save_pkl((X, y, train, valid), path=os.path.join(MODELS_PATH, model.get_name('data.pkl')))

# Fit models
model.fit_lgbm(X[train], y[train], X[valid], y[valid])
model.fit_catboost(X[train], y[train], X[valid], y[valid])
model.fit_neural(X[train], y[train], X[valid], y[valid])
model.fit_blender(X[valid], y[valid])

# Save model
model.save(os.path.join(MODELS_PATH, model.get_name()))
