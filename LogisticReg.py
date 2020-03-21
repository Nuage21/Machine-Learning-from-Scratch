import numpy as np
import pandas as pd


class LogisticReg:
    def __init__(self):
        self.X = None  # training data
        self.y = None  # labels
        self.learning_rate = 1e-2
        self.regularization_rate = 1e-2
        self.tol = 1e-3  # tolerance! amount of variation to error considered as convergence
        self.max_iter = 200  # max iterations to perform seeking convergence (respect to specified tolerance)
        self.batch_size = 10  # for mini-batch gradient descent
        self.fitted = False  # already fit ?
        self.cre_ = np.inf  # cross entropy error of the model
        self.weights = []  # Linear_Regression params
        self.verbose = 0  # display or not fitting progression

    def fit(self, X, y, learning_rate=1e-2, regularization_rate=1e-1, batch_size=10, max_iter=200,
            tol=1e-2, verbose=0):
        self.fitted = True
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        (n_samples, n_features) = X.shape
        self.weights = np.zeros((n_features,))  # init weights
        ex_error = np.inf  # check convergence with
        for iteration in range(max_iter):
            i = 0  # iterate over batches
            while i < n_samples:
                X_batch = X[i:min(i + batch_size, n_samples), :]
                y_predicted = self.sig(np.dot(X_batch, self.weights))  # E batch_size x n_features
                y_real = y[i:min(i + batch_size, n_samples)]
                self.weights[1:] *= 1 - (regularization_rate / n_samples)  # regularization part
                self.weights -= (learning_rate / n_samples) * np.dot(X_batch.T, y_predicted - y_real)
                i = i + batch_size  # next batch
            pred = self.sig(np.dot(X, self.weights))
            # print(pred)
            self.cre_ = -(np.dot(y, np.log(pred)) + np.dot(1 - y, np.log(1 - pred))) / n_samples
            if verbose:
                print('iteration ', iteration, ' error = ', self.cre_)
            if self.cre_ <= ex_error <= self.cre_ + tol:
                # if convergence before reaching max_iter
                if verbose:
                    print('Convergence!')
                return self
            elif iteration + 1 >= max_iter and verbose:
                print('Failure to converge after ', max_iter, ' iterations')
            ex_error = self.cre_
        return self

    def predict(self, X, prob=0, thresh=0.5):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
        pred = self.sig(np.dot(X, self.weights))
        if not prob:
            return np.int32(pred >= thresh)
        return pred

    def confusion_matrix(self, X, y, thresh=0.5):
        pred = self.predict(X, thresh=thresh)
        n_samples = len(y)
        p_sum = y + 2 * pred
        TP = np.count_nonzero(p_sum == 3)
        TN = np.count_nonzero(p_sum == 0)
        FP = np.count_nonzero(p_sum == 2)
        FN = n_samples - TP - TN - FP
        precision = TP / np.float32(TP + FP)
        recall = TP / np.float32(TP + FN)
        F1_score = 2 * precision * recall / np.float32(precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return np.array([[TP, FN], [FP, TN]]), precision, recall, F1_score, accuracy

    @staticmethod
    def sig(matrix):
        return 1 / (1 + np.exp(-matrix))
