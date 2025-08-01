import json
from sklearn.base import BaseEstimator, TransformerMixin

class JSONCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _count_json_array(self, s):
        try:
            s = s.replace("'", '"').replace("None", "null")
            parsed = json.loads(s)
            return len(parsed) if isinstance(parsed, list) else 0
        except:
            return 0

    def transform(self, X):
        X = X.copy()
        X["num_cast"] = X["cast"].apply(self._count_json_array)
        return X[["num_cast"]].values
