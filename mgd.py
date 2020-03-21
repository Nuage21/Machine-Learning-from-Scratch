import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            sigma = np.zeros((n_features, n_features))
            for i in range(n_samples):
                dt = np.dot((X[i, :] - mean).reshape(1, -1).T, (X[i, :] - mean).reshape(1, -1))
                sigma += dt
            sigma = sigma / n_samples
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


data = pd.read_csv('./datasets/mgd.csv')

X = data.iloc[:, 1:3].to_numpy(dtype=np.float32)
y = data.iloc[:, 3].to_numpy(dtype=np.int32)

X1 = X[:, 0]  # feature 1
X2 = X[:, 1]  # feature 2

X1_True = np.array([x1 for ind, x1 in enumerate(X1) if y[ind] == 1])  # x1's labeled true (1)
X2_True = np.array([x2 for ind, x2 in enumerate(X2) if y[ind] == 1])  # x2's labeled true (1)

X1_False = np.array([x1 for ind, x1 in enumerate(X1) if y[ind] == 0])  # x1's labeled true (0))
X2_False = np.array([x2 for ind, x2 in enumerate(X2) if y[ind] == 0])  # x2's labeled true (0)

# fig = plt.figure(figsize=(11, 6), dpi=80, facecolor='#e1e1e1')
# ax = fig.add_subplot(111)
# ax.set_facecolor('#1d1d1d')  # some dark background
#
# ax.scatter(X1_True, X2_True, marker='x', c='w', s=400)
# ax.scatter(X1_False, X2_False, marker='o', c='w', s=400)
#
# plt.ylim((-5, 6))  # y axis range
# plt.xlim((-1.5, 2.5))  # x axis range
#
# plt.title('Data-Points')
# plt.xlabel('Feature X1')
# plt.ylabel('Feature X2')
#
# plt.grid(color='#313131')
#
# plt.show()  # show plot

model = Mgd().fit(X, verbose=1)

th = 2.0570575e-02

# model.sigma[1, 1]
# model.sigma[1, 1] += 0.5
# print(model.sigma[1, 1])

cmat, precision, recall, f1_score, accuracy = model.confusion_matrix(X, y, thresh=th)

print(model.predict(X, prob=0, thresh=th))

print(cmat)
print('precision = ', precision)
print('recall = ', recall)
print('accuracy = ', accuracy)

x1_range = np.arange(-1.5, 2.5, 0.1)
x2_range = np.arange(-5, 6, 0.1)
X_range = []
for x1 in x1_range:
    for x2 in x2_range:
        X_range.append([x1, x2])
X_range = np.array(X_range)

y_range = model.predict(X_range, prob=1, thresh=th)
x1_range = X_range[:, 0]
x2_range = X_range[:, 1]

figure = plt.figure(dpi=200, facecolor='#e1e1e1')

ax_3d = figure.add_subplot(111, projection='3d')

ax_3d.scatter(x1_range, x2_range, y_range, marker='.', c='#1d1d1d', s=1)

mT = X1_True.shape[0]

X_range_true = np.zeros((mT, 2))
X_range_true[:, 0] = X1_True
X_range_true[:, 1] = X2_True

ax_3d.scatter(X1_True, X2_True, model.predict(X_range_true, prob=1), marker='x', c='r', s=100)


mF = X1_False.shape[0]

X_range_False = np.zeros((mF, 2))
X_range_False[:, 0] = X1_False
X_range_False[:, 1] = X2_False

ax_3d.scatter(X1_False, X2_False, model.predict(X_range_False, prob=1), marker='o', c='g', s=100)


plt.xlabel('feature x1')
plt.ylabel('feature x2')
plt.clabel('density')

plt.show()
