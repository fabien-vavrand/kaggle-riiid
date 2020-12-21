import os
import pickle
import numpy as np


PATH = r'C:\Users\chass\Kaggle\riiid'
INPUT_PATH = os.path.join(PATH, 'data')
MODELS_PATH = os.path.join(PATH, 'models')
SUBMIT_PATH = os.path.join(PATH, 'submit')
TEST_PATH = os.path.join(PATH, 'tests')
TUNE_PATH = os.path.join(PATH, 'tuning')


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
        'smoothing_min': None,
        'smoothing_value': 0.5
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
    'lgbm_params_raw': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.1,
        'verbosity': -1
    },
    'lgbm_params': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.99,
        'bagging_freq': 3,
        'num_leaves': 100,
        'min_child_samples': 20,
        'lambda_l1': 1.7e-3,
        'lambda_l2': 4.3e-7,
        'verbosity': -1
    },
    'mlp_params': {
        'layers': [512, 512, 512, 512, 512, 256, 128, 16],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 256
    }
}