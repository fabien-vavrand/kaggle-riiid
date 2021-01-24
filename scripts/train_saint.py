import os
from riiid.core.data import DataLoader, save_pkl
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH


configure_console_logging()

# Load data
loader = DataLoader(INPUT_PATH)
train, questions, lectures = loader.load_first_users(30000)

# Compute features
model = SaintModel(questions, lectures)
train = model.fit_transform(train)

# Create train and validation datasets
train, test = model.split_train_test(train)
train = model.create_features(train)
test = model.create_features(test)
X_train, y_train = model.create_dataset(train)
X_test, y_test = model.create_dataset(test)
save_pkl((X_train, y_train, X_test, y_test), os.path.join(MODELS_PATH, model.get_name('data.pkl')))

# Fit model
model.fit(X_train, y_train, X_test, y_test)
model.score(X_test, y_test)

# Save model
model.save(MODELS_PATH)
