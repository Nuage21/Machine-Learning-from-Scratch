import numpy as np


class LinearReg:
    def __init__(self):
        self.X = None  # training data
        self.y = None  # labels
        self.learning_rate = 1e-2
        self.regularization_rate = 1e-1
        self.tol_ = 1e-3  # tolerance! amount of variation to error considered as convergence
        self.batch_size = 10  # for mini-batch gradient descent
        self.solver = 'sgd'  # stochastic gradient descent by default
        self.fit = False  # already fit ?
        self.var_ = np.inf  # sum of squared errors of the model

    def fit(self, X, y, solver='sgd', learning_rate=1e-2, regularization_rate=1e-1, batch_size=10):
        iter = 0
        while iter <

    def predict(self, X):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
        pass
