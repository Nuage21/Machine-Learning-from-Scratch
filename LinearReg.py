import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearReg:
    def __init__(self):
        self.X = None  # training data
        self.y = None  # labels
        self.learning_rate = 1e-2
        self.regularization_rate = 1e-2
        self.tol = 1e-3  # tolerance! amount of variation to error considered as convergence
        self.max_iter = 200  # max iterations to perform seeking convergence (respect to specified tolerance)
        self.batch_size = 10  # for mini-batch gradient descent
        self.fitted = False  # already fit ?
        self.mae_ = np.inf  # mean absolute error of the model
        self.weights = []  # Linear_Regression params
        self.verbose = 0

    def fit(self, X, y, learning_rate=1e-2, regularization_rate=1e-1, batch_size=10, max_iter=200,
            tol=1e-2, verbose=0):
        # will fit the model and return self
        # ...
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
                y_predicted = np.dot(X_batch, self.weights)  # E batch_size x n_features
                y_real = y[i:min(i + batch_size, n_samples)]
                self.weights[1:] *= 1 - (regularization_rate / n_samples)  # regularization part
                self.weights -= (learning_rate / n_samples) * np.dot(X_batch.T, y_predicted - y_real)
                i = i + batch_size  # next batch
            # compute new cost
            self.mae_ = (np.sum((np.dot(X, self.weights) - y) ** 2) + (
                    regularization_rate * np.sum(np.insert(self.weights[1:], 0, 0) ** 2))) / (2 * n_samples)
            if verbose:
                print('iteration ', iteration, ' error = ', self.mae_)
            if self.mae_ + tol >= ex_error:
                # if convergence before reaching max_iter
                if verbose:
                    print('Convergence!')
                return self
            elif iteration + 1 >= max_iter and verbose:
                print('Failure to converge after ', max_iter, ' iterations')
            ex_error = self.mae_
        return self

    def score(self, X, y, tol=1):
        # get accuracy of prediction over X & y
        # score the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
        y_pred = np.dot(X, self.weights)
        d_avg = np.sum(np.abs(y_pred - y)) / len(y)  # average distance from correct prediction
        return np.exp(- (np.log(2) * d_avg) / tol), d_avg

    def predict(self, X):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
        return np.dot(X, self.weights)
