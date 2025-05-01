from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

class DT_Estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        
    def fit(self, X, y):
        self.estimator_ = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split,criterion=self.criterion,
        random_state=self.random_state).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)
        
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)