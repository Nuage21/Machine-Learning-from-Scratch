import numpy as np


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
                self.weights = ((1 - (regularization_rate / n_samples)) * np.insert(self.weights[1:], 0, 0)) - (
                        learning_rate / n_samples) * np.dot(X_batch.T, y_predicted - y_real)
                i = i + batch_size  # next batch
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

    def predict(self, X):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')
        return np.dot(X, self.weights)


X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
y = np.array([2, 4, 6, 8])

model = LinearReg().fit(X, y, verbose=1, max_iter=400, tol=0, regularization_rate=1e-1, batch_size=4, learning_rate=0.01)

P = model.predict(np.array([[1, 7], [1, 8], [1, 9]]))

print(P)