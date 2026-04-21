import numpy as np

class LogScaler:
    def __init__(self, target_range=(0, 1)):
        self.t_min, self.t_max = target_range
        # Learned attributes get trailing underscore by sklearn convention
        self.log_min_ = None
        self.log_max_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, data):
        data = np.asarray(data)
        self.data_min_ = np.min(data)
        self.data_max_ = np.max(data)
        self.log_min_ = np.log1p(self.data_min_)
        self.log_max_ = np.log1p(self.data_max_)
        return self

    def transform(self, data):
        self._check_is_fitted()
        data = np.asarray(data)
        denom = self.log_max_ - self.log_min_
        if denom == 0:
            return np.full_like(data, self.t_min, dtype=float)
        range_width = self.t_max - self.t_min
        return self.t_min + ((np.log1p(data) - self.log_min_) / denom) * range_width

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, scaled):
        self._check_is_fitted()
        scaled = np.asarray(scaled)
        denom = self.log_max_ - self.log_min_
        if denom == 0:
            return np.full_like(scaled, self.data_min_, dtype=float)
        range_width = self.t_max - self.t_min
        log_val = (scaled - self.t_min) / range_width * denom + self.log_min_
        return np.expm1(log_val)

    def _check_is_fitted(self):
        if self.log_min_ is None:
            raise RuntimeError("call fit() before transform()")