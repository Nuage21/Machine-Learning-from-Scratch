import numpy as np

class Mgd:

    def __init__(self):
        self.sigma = None  # covariance matrix
        self.thresh = 1e-3  # threshold on probability
        self.mean = None  # mean vector of training data
        self.sigma_inv = None  # sigma^-1
        self.sigma_det = None  # |sigma|
        self.fitted = 0  # model fitted ?
        self.n_features = 0

    def fit(self, X, sigma=None, mean=None, verbose=0):
        (n_samples, n_features) = X.shape
        self.n_features = n_features
        if mean is None:
            mean = np.mean(X, axis=0)
            if verbose:
                print('Mean vector of training data successfully computed!')
        if sigma is None:
            X_mean = X - np.mean(X, axis=0)
            sigma = np.dot(X_mean.T, X_mean) / n_samples
            if verbose:
                print('Covariance matrix of training data successfully computed!')

        self.sigma = sigma
        self.mean = mean
        # needed
        self.sigma_inv = np.linalg.pinv(sigma)
        if verbose:
            print('Covariance inverse matrix of training data successfully computed!')
        self.sigma_det = np.linalg.det(sigma)
        if self.sigma_det == 0:
            raise ZeroDivisionError('|Sigma| = 0; can\'t train')
        if verbose:
            print('Covariance matrix determinant of training data successfully computed!')
        self.fitted = 1
        return self

    def predict(self, X, prob=0, thresh=1e-3):
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
        (m, _) = X.shape
        p_res = np.zeros((m,))
        for i in range(m):
            Xc = X[i, :] - self.mean
            inside_exp = np.dot(Xc.T, self.sigma_inv)
            p_res[i] = np.exp(-0.5 * np.dot(inside_exp, Xc)) / (
                    np.sqrt(self.sigma_det) * np.power(np.pi, self.n_features))
        if prob:
            return p_res
        return np.int32(p_res <= thresh)

    def confusion_matrix(self, X, y, thresh=0.5):
        self.thresh = thresh  # save
        pred = self.predict(X, thresh=thresh)
        n_samples = len(y)
        p_sum = y + 2 * pred
        TP = np.count_nonzero(p_sum == 3)
        TN = np.count_nonzero(p_sum == 0)
        FP = np.count_nonzero(p_sum == 2)
        FN = n_samples - TP - TN - FP
        precision = TP / np.float32(TP + FP)
        recall = TP / np.float32(TP + FN)

        F1_score = 0
        if precision + recall != 0:
            F1_score = 2 * precision * recall / np.float32(precision + recall)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return np.array([[TP, FN], [FP, TN]]), precision, recall, F1_score, accuracy
