import numpy as np


def draw_params():
    layers_sizes = [8 + 4 * n for n in range(20)]

    return {
        'question_embedding': {
            'n_clusters': np.random.randint(30, 100),
            'embedding_size': int(np.random.choice(layers_sizes, 1)[0]),
            'window': np.random.randint(1, 7),
            'min_count': np.random.randint(1, 10),
            'sg': np.random.randint(0, 2),
            'iter': np.random.randint(5, 50)
        },
        'answers_embedding': {
            'n_fold': 5,
            'embedding_size': int(np.random.choice(layers_sizes, 1)[0]),
            'window': np.random.randint(1, 7),
            'min_count': np.random.randint(1, 10),
            'sg': np.random.randint(0, 2),
            'iter': np.random.randint(5, 50),
        },
        'answer_encoder': {
            'smoothing_min': np.random.uniform(0, 8),
            'smoothing_value': np.random.uniform(0.5, 3),
        },
        'score_encoder': {
            'smoothing_min': np.random.uniform(0, 8),
            'smoothing_value': np.random.uniform(0.5, 3),
            'noise': 0.005
        },
        'user_score_encoder': {
            'smoothing_min': np.random.uniform(0, 8),
            'smoothing_value': np.random.uniform(0.5, 3),
        },
        'user_content_score_encoder': {
            'smoothing_min': np.random.uniform(0, 8),
            'smoothing_value': np.random.uniform(0.5, 3),
        },
        'user_rolling_score_encoder': {
            'smoothing_min': np.random.uniform(0, 8),
            'smoothing_value': np.random.uniform(0.5, 3),
            'rolling': np.random.randint(5, 100)
        },
        'user_weighted_score_encoder': {
            'smoothing_min': np.random.uniform(0, 8),
            'smoothing_value': np.random.uniform(0.5, 3)
        }
    }
