import os
from riiid.core.data import DataLoader
from riiid.saint.model import SaintModel
from riiid.utils import configure_console_logging
from riiid.config import INPUT_PATH, MODELS_PATH


configure_console_logging()

loader = DataLoader(INPUT_PATH)
train, questions, lectures = loader.load_first_users(10000)

model = SaintModel(questions, lectures)
train, test = model.fit(train)
model.train(train, test)
model.save(os.path.join(MODELS_PATH, model.get_name()))
