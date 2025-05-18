from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

class LR_Estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, penalty='l2', max_iter=100, random_state=None):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        self.estimator_ = LogisticRegression(C=self.C,penalty=self.penalty,max_iter=self.max_iter,random_state=self.random_state).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)
        
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)