from __future__ import annotations
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC

class SVMWithRFF:
    """Linear SVM on random Fourier features (RBF approximation)."""
    def __init__(self, n_components=300, gamma=0.5, C=1.0, random_state=42):
        self.rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
        self.clf = LinearSVC(C=C, class_weight='balanced', dual=True)
        self.is_fit = False

    def fit(self, X, y):
        Z = self.rbf.fit_transform(X)
        self.clf.fit(Z, y)
        self.is_fit = True
        return self

    def decision_function(self, X):
        Z = self.rbf.transform(X)
        return self.clf.decision_function(Z)

    def predict(self, X):
        Z = self.rbf.transform(X)
        return self.clf.predict(Z)
