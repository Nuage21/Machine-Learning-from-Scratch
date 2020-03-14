import numpy as np
from matplotlib import pyplot as plt

total_samples = 200

X = np.random.random_sample((total_samples, 2)) * 100


def plot2d(X_input, format='rx'):
    X_f1 = X_input[:, 0]
    X_f2 = X_input[:, 1]
    plt.plot(X_f2, X_f1, format, color='#1d1d1d')


def init_centroids(input, k):
    cent_coord_list = []  # a list of coords (tuple)
    cent_index_list = []  # cent indices for unique selection
    (m, n) = input.shape
    for c in range(k):
        rd = np.random.randint(0, m, dtype=np.int32)
        while rd in cent_index_list:
            rd = np.random.randint(0, m, dtype=np.int32)
        c_coord = input[rd, :]
        cent_coord_list.append(c_coord)
        cent_index_list.append(rd)
    return np.array(cent_coord_list)


def fit(input, centroids, max_iter=-1):
    # will return: [final centroids], map-matrix [ [sample-index, centroid-index], ....]
    k = len(centroids)
    (m, n) = input.shape
    ret = np.zeros((m,), dtype=np.int32)
    infinit_loop = max_iter <= 0
    p = 0
    last_attributed_centroids = np.array([])
    while True:
        for i in range(m):
            point = input[i, :]
            attributed_cent = 0
            attributed_norm = np.linalg.norm(point - centroids[0, :])
            for j in np.arange(k - 1) + 1:
                cent = centroids[j, :]
                norm = np.linalg.norm(point - cent)  # distance
                if norm < attributed_norm:
                    attributed_norm = norm
                    attributed_cent = j
            ret[i] = attributed_cent
        # now edit each centroids to the mean of its followers
        for i in range(k):
            # array of samples-indexes following the i'th centroid
            ind_followers = np.array([ind for ind, cent in enumerate(ret) if cent == i])
            # matrix of samples(real-values) following the i'th centroid
            followers = np.array([sample for ind, sample in enumerate(input) if (ind in ind_followers)])
            mean_vec = np.mean(followers, axis=0)  # average
            centroids[i] = mean_vec
        p = p + 1
        if not infinit_loop:
            if p >= max_iter:
                break
        elif np.size(last_attributed_centroids) > 0:
            # if no max_iter specified then run till getting no centroids change over 2-iterations
            cmp = np.equal(ret, last_attributed_centroids)  # compare attributed centroids one-by-one
            if np.product(cmp) == 1:  # perform and-logic
                break
        last_attributed_centroids = ret
    return centroids, ret

K = 3  # number of clusters

random_centroids = init_centroids(X, K)
centroids, attr = fit(X, random_centroids)


def get_clusters(input_data, attr_vec, k):
    # return a list containing the input_data subdivided into clusters (specified in attr_vec)
    clusters = []
    for i in range(k):
        X_ci = np.array([row for ind, row in enumerate(input_data) if attr_vec[ind] == i])
        clusters.append(X_ci)
    return clusters


clusters = get_clusters(X, attr, K)

plot2d(clusters[0], format='ro')
plot2d(clusters[1], format='rx')
plot2d(clusters[2], format='r-')

plt.show()
