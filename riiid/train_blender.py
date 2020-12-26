import os
import logging
from riiid.core.data import load_pkl
from riiid.core.model import RiiidModel
from riiid.core.neural import NeuralModel
from riiid.utils import configure_console_logging
from riiid.config import PATH, INPUT_PATH, MODELS_PATH, PARAMS


configure_console_logging()

model: RiiidModel = RiiidModel.load(os.path.join(MODELS_PATH, 'model_20201222_175136.zip'))
X, y, train, valid = load_pkl(os.path.join(MODELS_PATH, model.get_name('data.pkl')))

add_neural = os.path.exists(os.path.join(MODELS_PATH, model.get_name('neural.zip')))
if add_neural:
    logging.info('Adding neural model')
    neural_model = NeuralModel.load(os.path.join(MODELS_PATH, model.get_name('neural.zip')))
    model.models.append({'model': neural_model})

model.fit_blender(X[valid], y[valid])

if add_neural:
    model.models.pop(-1)

model.save(os.path.join(MODELS_PATH, model.get_name()))
