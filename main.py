import numpy as np
from matplotlib import pyplot as plt

total_samples = 70

X = np.random.random_sample((total_samples, 2)) * 10


def plot2d(X_input):
    X_f1 = X_input[:, 0]
    X_f2 = X_input[:, 1]
    plt.plot(X_f2, X_f1, 'rx', color='#1d1d1d')
    plt.show()


plot2d(X)

K = 2  # number of clusters


def init_centroids(input, k):
    cent_coord_list = []  # a list of coords (tuple)
    cent_index_list = []  # cent indices for unique selection
    (m, n) = input.shape
    for c in range(k):
        rd = np.random.randint(0, m, dtype=np.int32)
        while rd in cent_index_list:
            rd = np.random.randint(0, m, dtype=np.int32)
        c_coord = tuple(input[rd, :])
        cent_coord_list.append(c_coord)
        cent_index_list.append(rd)
    return cent_coord_list


coords = init_centroids(X, 3)

print(coords)
