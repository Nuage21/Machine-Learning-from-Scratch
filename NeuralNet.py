import numpy as np
from sklearn.neural_network import MLPClassifier


class NeuralNet:
    def __init__(self, nn_type='reg', hidden_layer_sizes=(5,), learning_rate=1e-3, regularization_rate=1e-4,
                 activation='relu', tol=1e-3, max_iter=300):
        self.hidden_layer_size = hidden_layer_sizes
        self.nn_type = nn_type
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.activation = activation
        self.tol = tol
        self.max_iter = max_iter
        self.weighs = []
        self.fitted = 0

    def fit(self, X, y, verbose=True):
        self.fitted = True
        m, n = X.shape
        out_size = 1
        if y.ndim == 2:
            out_size = y.shape[1]
        self.init_weighs(input_size=n, output_size=out_size)

    def predict(self, X, prob=0):
        self.check_for_error()
        if prob:
            return self.feed_forward(X, return_history=0).T

    def score(self, X, y):
        self.check_for_error()

    def feed_forward(self, X, return_history=0):
        # returns the list of layers after feed forward
        self.check_for_error()
        y_p = X.T
        history = []
        for i, w in enumerate(self.weighs):
            y_p = np.dot(w, np.insert(y_p, [0], 1, axis=0))
            if return_history and i < (len(self.weighs) - 1):
                y_p = self.activate(y_p)
                history.append(y_p)
        if 'c' in self.nn_type:  # classifier
            if y_p.ndim == 1:
                y_p = self.sig(y_p)
            else:
                y_p = np.exp(y_p)
                y_p /= np.sum(y_p, axis=0)
        if return_history:
            history.append(y_p)
            return history
        return y_p

    def init_weighs(self, input_size, output_size):
        # Xavier init weighs if 'tanh' activation and He if 'ReLu'
        layer_sizes = [input_size] + list(self.hidden_layer_size) + [output_size]
        n_layer = len(layer_sizes)
        # special weighs init
        c = 2  # He init
        if self.activation == 'tanh':  # Xavier init
            c = 1
        cf = lambda x: np.sqrt(c / x)
        for i in range(n_layer - 1):
            n_rows = layer_sizes[i + 1]
            n_cols = layer_sizes[i] + 1
            self.weighs.append(np.random.randn(n_rows, n_cols) * cf(n_cols))

    def check_for_error(self):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')

    def activate(self, layer):
        f = getattr(self, self.activation)
        return f(layer)

    @staticmethod
    def sig(matrix):
        return 1 / (1 + np.exp(-matrix))

    @staticmethod
    def relu(matrix):
        matrix[matrix <= 0] = 0
        return matrix

    @staticmethod
    def tanh(matrix):
        tmp = np.exp(-2 * matrix)
        return (1 - tmp) / (1 + tmp)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1], [1, 1], [1, 2], [0, 3]])

model = NeuralNet(nn_type='classifier', hidden_layer_sizes=(3, 2), regularization_rate=0, activation='sig')
model.fit(X, y)
