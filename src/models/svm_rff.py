from __future__ import annotations
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

class SVMWithRFF:
    """Linear SVM on random Fourier features (RBF approximation) with optional probability calibration."""
    def __init__(self, n_components=300, gamma=0.5, C=1.0, random_state=42, calibrate: bool = True, cv: int = 3):
        self.rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
        base_svm = LinearSVC(C=C, class_weight='balanced', dual=True)
        self.calibrate = calibrate
        self.cv = cv
        self._base = base_svm
        self.clf = None
        self.is_fit = False

    def fit(self, X, y):
        Z = self.rbf.fit_transform(X)
        if self.calibrate:
            self.clf = CalibratedClassifierCV(self._base, method='sigmoid', cv=self.cv)
        else:
            self.clf = self._base
        self.clf.fit(Z, y)
        self.is_fit = True
        return self

    def decision_function(self, X):
        Z = self.rbf.transform(X)
        # CalibratedClassifierCV exposes predict_proba/decision_function depending on base; use decision_function if available
        if hasattr(self.clf, 'decision_function'):
            return self.clf.decision_function(Z)
        # fallback to probability of positive class mapped to logit-like scores
        if hasattr(self.clf, 'predict_proba'):
            proba = self.clf.predict_proba(Z)[:, 1]
            # Convert to logit-ish score centered at 0 for downstream thresholding consistency
            eps = 1e-8
            return np.log((proba + eps) / (1 - proba + eps))
        raise AttributeError("Classifier has neither decision_function nor predict_proba")

    def predict(self, X):
        Z = self.rbf.transform(X)
        return self.clf.predict(Z)

    def predict_proba(self, X):
        Z = self.rbf.transform(X)
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(Z)
        # approximate probabilities from decision function via sigmoid
        scores = self.decision_function(X)
        proba_pos = 1.0 / (1.0 + np.exp(-scores))
        return np.vstack([1 - proba_pos, proba_pos]).T
