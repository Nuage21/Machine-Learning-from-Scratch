import numpy as np


class PCA:
    def __int__(self):
        self.e_vals = None  # vector of eigen values
        self.n_features = -1
        self.n_samples = -1
        self.fitted = 0  # model fitted ?

    def fit(self, X):
        (self.n_samples, self.n_features) = X.shape
        sigma = np.dot(X.T, X) / self.n_features  # covariance matrix dims = n x n
        # compute eigen values
        self.e_vals = np.linalg.eigvals(sigma)
        self.fitted = 1

    def transform(self, X, n_comp):
        self.check_if_fitted()
        return np.dot(X, self.e_vals[0:n_comp] * np.eye(n_comp))

    def recover(self, X_red):
        self.check_if_fitted()
        _, n_comp = X_red.shape
        return np.dot(X_red, (self.e_vals[0:n_comp] * np.eye(n_comp)).T)

    def compute_cov(self, n_comp):
        self.check_if_fitted()
        squared_eig = self.e_vals ** 2
        sum_comp = np.sum(squared_eig[0:n_comp])
        sum_all = np.sum(squared_eig)
        return 1 - (sum_comp / sum_all)

    def check_if_fitted(self):
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
