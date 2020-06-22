import numpy as np


class NeuralNet:
    def __init__(self, nn_type, hidden_layer_sizes=(5,), activation='relu'):
        self._supported_activations = ['relu', 'tanh', 'sig']
        # check params
        if nn_type not in ['reggressor', 'sigmoid_classifier', 'softmax_classifier']:
            raise NotImplementedError(
                f'> {nn_type}: nn_type should either be \'sigmoid_classifier\' (for 2C classification), \'softmax_classifier\' or \'reggressor\' (for regression)')
        if activation not in self._supported_activations:
            raise NotImplementedError(
                f'{activation} activation function is not supported, implemented are {self._supported_activations}')
        for l in hidden_layer_sizes:
            if not isinstance(l, int):
                raise TypeError(f'> {hidden_layer_sizes}, a layer size should be an integer')
            if l <= 0:
                raise ValueError(f'> {hidden_layer_sizes}, a layer size should be a positive integer')

        self.hidden_layer_size = hidden_layer_sizes
        self.nn_type = nn_type
        self.learning_rate = 1e-3
        self.regularization_rate = 1e-4
        self.activation = activation
        self.tol = 1e-3
        self.max_iter = 300
        self.weighs = []
        self.fitted = 0
        self.error = np.inf
        self.converged_ = False
        self.n_check_no_change = 25
        self.batch_size = 10
        self.stch_iter_err_check = 1
        self.verbose = False
        self._is_classifier = False
        if 'c' in nn_type:
            self._is_classifier = True

    def fit(self, X, y, learning_rate=1e-3, reg='l2', regularization_rate=1e-4, tol=1e-3, max_iter=300,
            n_check_no_change=25, batch_size=10, stch_iter_err_check=1, warm_start=False, verbose=False):
        if reg not in ['l1', 'l2']:
            raise NotImplementedError(f'> {reg}: supported regularizations are \'l1\' and \'l2\'')
        if warm_start and not self.fitted:
            raise RuntimeError('Can\'t warm start if model is not fitted yet')
        # save (for warm start ?)
        self.fitted = True
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.n_check_no_change = n_check_no_change
        self.batch_size = batch_size
        self.stch_iter_err_check = stch_iter_err_check
        self.verbose = verbose
        # begin
        m, n = X.shape
        out_size = 1
        if y.ndim == 2:
            out_size = y.shape[1]
        if not warm_start:
            self._init_weighs(input_size=n, output_size=out_size)
        error_calculator = self._avg_squared_error
        if self._is_classifier:
            error_calculator = self._avg_cross_entropy
        deactivator = self._get_deactivator()
        ex_error = np.inf
        no_change_counter = 0
        n_iters = 0
        for k in range(max_iter):
            j = 0
            while j < m:
                X_batch = X[j:min(j + batch_size, m), :]
                history = [X_batch.T] + self.feed_forward(X_batch, return_history=1)
                y_p = history[-1]
                if n_iters % stch_iter_err_check == 0:
                    self.error = error_calculator(y, self.predict(X, prob=1), reg)
                    if verbose:
                        print(f'iteration {k}, {error_calculator.__name__[1:]} = {self.error}')
                    # check for convergence
                    if self.error <= ex_error <= self.error + tol:
                        # if convergence before reaching max_iter
                        no_change_counter += 1
                        if no_change_counter >= n_check_no_change:
                            self.converged_ = 1
                            if verbose:
                                print('Convergence!')
                            return self
                    else:
                        no_change_counter = 0  # reset
                n_lay = len(history)
                sigma_nxt = y_p - y[j:min(j + batch_size, m)]
                for i in np.arange(n_lay - 2, -1, -1):
                    act_i = np.insert(history[i], [0], 1, axis=0)  # add bias line (1)
                    grad = np.dot(sigma_nxt, act_i.T) / m
                    # compute weigh regularization respect to specified (l1 | l2)
                    w_reg = (self.weighs[i])[:, 1:].copy()  # if reg = 'l2'
                    if reg == 'l1':
                        w_reg[w_reg > 0] = 1  # l1 gradient
                        w_reg[w_reg < 0] = -1  # l1 gradient
                    grad[:, 1:] += (regularization_rate * w_reg) / m
                    self.weighs[i] -= (learning_rate * grad)
                    if i == 0:
                        break
                    sigma_nxt = np.dot((self.weighs[i])[:, 1:].T, sigma_nxt) * deactivator(act_i[1:, :])
                ex_error = self.error
                j += batch_size
                n_iters += 1
        # if arrived here than no convergence
        self.error = error_calculator(y, self.predict(X, prob=1), reg)  # update error
        print(f'last iteration, {error_calculator.__name__[1:]} = {self.error}')
        print(f'Failure to converge! final {error_calculator.__name__[1:]} = {self.error}')
        return self

    def predict(self, X, prob=0, thresh=0.5):
        self._check_for_error()
        y = self.feed_forward(X, return_history=0)[0].T
        if prob or not self._is_classifier:
            return y
        return np.int32(y >= thresh)

    def score(self, X, y):
        self._check_for_error()

    def feed_forward(self, X, return_history=0):
        # returns the list of layers after feed forward
        self._check_for_error()
        activator = self._get_activator()
        y_p = X.T
        history = []
        for i, w in enumerate(self.weighs):
            y_p = np.dot(w, np.insert(y_p, [0], 1, axis=0))
            if i < (len(self.weighs) - 1):
                y_p = activator(y_p)
                if return_history:
                    history.append(y_p)
        if self._is_classifier:
            if y_p.shape[0] == 1:
                y_p = self.sig(y_p)
        if return_history:
            history.append(y_p)
            return history
        return y_p

    def _init_weighs(self, input_size, output_size):
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
            w = np.random.randn(n_rows, n_cols - 1) * cf(n_cols - 1)
            w = np.insert(w, [0], 1, axis=1)
            self.weighs.append(w)

    def _avg_cross_entropy(self, y, y_p, reg='l2'):
        # compute cross entropy error with regularization rate (l1 or l2)
        ce = -np.sum(y * np.log(y_p)) - np.sum((1 - y) * np.log(1 - y_p))  # average cross entropy
        regularizator = self._compute_regularization_rate(reg)
        return (ce + regularizator) / len(y)

    def _avg_squared_error(self, y, y_p, reg='l2'):
        sqr_error = np.sum((y_p - y) ** 2)
        regularizator = self._compute_regularization_rate(reg)
        return (sqr_error + regularizator) / len(y)

    def _compute_regularization_rate(self, reg='l2'):
        regularizator = 0
        if reg == 'l2':
            for w in self.weighs:
                regularizator += np.sum(w[:, 1:] ** 2)
            regularizator *= (self.regularization_rate / 2)
        else:  # l1
            for w in self.weighs:
                regularizator += np.sum(np.abs(w[:, 1:]))
            regularizator *= self.regularization_rate
        return regularizator

    def _check_for_error(self):
        # predict the output for the matrix of inputs
        if not self.fitted:
            raise RuntimeError('Model not fitted yet!')

    def _get_activator(self):
        f = getattr(self, self.activation)
        return f

    def _get_deactivator(self):
        grd_fnc = self.activation + '_gradient'  # convention
        f_prime = getattr(self, grd_fnc)
        return f_prime

    @staticmethod
    def sig(matrix):
        return 1 / (1 + np.exp(-matrix))

    @staticmethod
    def relu(matrix):
        tmp = matrix.copy()
        tmp[tmp <= 0] = 0
        return tmp

    @staticmethod
    def tanh(matrix):
        tmp = np.exp(-2 * matrix)
        return (1 - tmp) / (1 + tmp)

    @staticmethod
    def sig_gradient(matrix):
        return matrix * (1 - matrix)

    @staticmethod
    def tanh_gradient(matrix):
        return 1 - matrix.copy() ** 2

    @staticmethod
    def relu_gradient(matrix):
        tmp = matrix.copy()
        tmp[tmp > 0] = 1  # gradient
        return tmp
