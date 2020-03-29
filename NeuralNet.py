import numpy as np


class NeuralNet:
    def __init__(self, type='reg', hidden_layer_sizes=(5,), learning_rate=1e-3, regularization_rate=1e-4,
                 activation='relu', tol=1e-3, max_iter=300):
        self.hidden_layer_size = hidden_layer_sizes
        self.type = type
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.activation = activation
        self.tol = tol
        self.max_iter = max_iter
        self.weighs = []
        self.fitted = 0

    def init_weighs(self, input_size, output_size):
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

    def fit(self, X, y, verbose=True):
        m, n = X.shape
        self.init_weighs(input_size=n, output_size=len(y))

    def predict(self, X, prob=0):
        self.check_for_error()

    def score(self, X, y):
        self.check_for_error()

    def feed_forward(self, X, weighs):
        pass

    def activate(self, layer):
        f = getattr(self, self.activation)
        return f(layer)

    def check_for_error(self):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')

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
