import os
import numpy as np
from sklearn.metrics import roc_auc_score

from riiid.core.data import save_pkl, load_pkl
from riiid.core.model import RiiidModel
from riiid.core.neural import NeuralModel
from riiid.utils import configure_console_logging
from riiid.config import MODELS_PATH, PARAMS


configure_console_logging()

model = RiiidModel.load(os.path.join(MODELS_PATH, 'model_20201224_172142.zip'))
X, y, train, valid = load_pkl(os.path.join(MODELS_PATH, model.get_name('data.pkl')))
X = X.to_numpy(dtype=np.float32)

# Train and save model
nn = NeuralModel(PARAMS['mlp_params'])
nn.fit(X[train], y[train], X[valid], y[valid])
nn.save(os.path.join(MODELS_PATH, model.get_name('neural.zip')))

"""
# Load model
nn: NeuralModel = NeuralModel.load(os.path.join(MODELS_PATH, model.get_name('neural.zip')))
y_hat = nn.predict(X[valid])

print(roc_auc_score(y[valid], y_hat))
"""