import os
import numpy as np


APPS_PATH = os.getenv('APPS_PATH', r'C:\Users\{}\app'.format(os.getlogin()))
PATH = os.getenv('RIIID_PATH', r'C:\Users\{}\Kaggle\riiid'.format(os.getlogin()))
SRC_PATH = os.path.join(PATH, 'kaggle-riiid')
INPUT_PATH = os.path.join(PATH, 'data')
MODELS_PATH = os.path.join(PATH, 'models')
SUBMIT_PATH = os.path.join(PATH, 'submit')
TEST_PATH = os.path.join(PATH, 'tests')
TUNE_PATH = os.path.join(PATH, 'tuning')

for path in [MODELS_PATH, SUBMIT_PATH, TEST_PATH, TUNE_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)


FLOAT_DTYPE = np.float32
ACTIVATE_CACHE = True


PARAMS = {
    'n_fold': 5,
    'question_embedding': {
        'n_clusters': 67,
        'embedding_size': 56,
        'window': 5,
        'min_count': 1,
        'sg': 0,
        'iter': 15,
        'workers': 1
    },
    'answers_embedding': {
        'n_fold': 5,
        'embedding_size': 68,
        'window': 3,
        'min_count': 5,
        'sg': 1,
        'iter': 20,
        'workers': 1
    },
    'answer_encoder': {
        'smoothing_min': 5,
        'smoothing_value': 1,
    },
    'score_encoder': {
        'smoothing_min': 5,
        'smoothing_value': 1,
        'noise': 0.001
    },
    'score_encoder_2': {
        'smoothing_min': 20,
        'smoothing_value': 5,
        'noise': 0.002
    },
    'user_score_encoder': {
        'smoothing_min': 5,
        'smoothing_value': 2
    },
    'user_content_score_encoder': {
        'smoothing_min': None
    },
    'user_rolling_score_encoder': {
        'smoothing_min': 2.5,
        'smoothing_value': 1,
        'rolling': 20
    },
    'user_weighted_score_encoder': {
        'smoothing_min': 2,
        'smoothing_value': 1,
    },
    'lgbm_params': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'num_leaves': 500,
        'min_child_samples': 1100,
        'verbosity': -1
    },
    'mlp_params': {
        'layers': [1024, 1024, 1024, 1024, 512, 256, 128, 16],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 256
    }
}
