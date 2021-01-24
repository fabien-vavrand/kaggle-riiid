import scipy
import logging
import numpy as np

from sklearn.preprocessing import PowerTransformer, StandardScaler


class ScalingTransformer:
    """
    Custom scaler which applies:
        - a power transformer on skewed features
        - a standard scaler on other features
    """
    def __init__(self, min_unique_values=5, skewness_threshold=1, max_rows=10_000_000):
        self.min_unique_values = min_unique_values
        self.skewness_threshold = skewness_threshold
        self.max_rows = max_rows
        self.rows = None
        self.columns = None
        self.standard_features = None
        self.skewed_features = None
        self.standard_scaler = None
        self.power_scaler = None

    def fit(self, X, y=None):
        logging.info('- Fit scaling transformer')
        self.rows, self.columns = X.shape
        self.standard_features = []
        self.skewed_features = []
        self.standard_scaler = StandardScaler()
        self.power_scaler = PowerTransformer()

        for i in range(self.columns):
            n_uniques = len(np.unique(X[:, i]))
            if n_uniques <= self.min_unique_values:
                self.standard_features.append(i)
            else:
                skewness = scipy.stats.skew(X[:, i])
                if skewness > self.skewness_threshold:
                    self.skewed_features.append(i)
                else:
                    self.standard_features.append(i)

        self.standard_features = np.array(self.standard_features)
        self.skewed_features = np.array(self.skewed_features)
        logging.info('{} standard features'.format(len(self.standard_features)))
        logging.info('{} skewed features'.format(len(self.skewed_features)))

        if self.rows > self.max_rows:
            X = X.sample(n=self.max_rows)

        self.standard_scaler.fit(X[:, self.standard_features])
        self.power_scaler.fit(X[:, self.skewed_features])
        return self

    def transform(self, X):
        return np.hstack([
            self.standard_scaler.transform(X[:, self.standard_features]),
            self.power_scaler.transform(X[:, self.skewed_features]),
        ])
