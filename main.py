import numpy as np
from matplotlib import pyplot as plt


class kMeans:
    def __init__(self):
        self.X = None  # data
        self.K = 2  # number of clusters
        self.max_iter = -1  # number of clusters
        self.fitted = False  # fit called ?
        self.centroids = None
        self.aff_ = None

    def fit(self, X, K=2, max_iter=-1):
        # will return: [final centroids], array of (n_samples, ) containing attributed centroid's index
        self.X = X
        self.K = K
        self.max_iter = max_iter
        # let's begin with randomly init centroids
        cent_coord_list = []  # a list of coords (tuple)
        cent_index_list = []  # cent indices for unique selection
        (m, n) = X.shape
        for c in range(K):
            rd = np.random.randint(0, m, dtype=np.int32)
            while rd in cent_index_list:
                rd = np.random.randint(0, m, dtype=np.int32)
            c_coord = X[rd, :]
            cent_coord_list.append(c_coord)
            cent_index_list.append(rd)
        self.centroids = np.array(cent_coord_list)
        ret = np.zeros((m,), dtype=np.int32)
        # if max_iter <= 0 then iterate till no change in attributed centroids over two successive iterations
        infinite_loop = max_iter <= 0
        p = 0  # max_iter arrived checker
        last_attributed_centroids = np.array([])
        while True:
            for i in range(m):
                point = X[i, :]
                attributed_cent = 0
                attributed_norm = np.linalg.norm(point - self.centroids[0, :])
                for j in np.arange(K - 1) + 1:
                    cent = self.centroids[j, :]
                    norm = np.linalg.norm(point - cent)  # distance
                    if norm < attributed_norm:
                        attributed_norm = norm
                        attributed_cent = j
                ret[i] = attributed_cent
            # now edit each centroids to the mean of its followers
            for i in range(K):
                # array of samples-indexes following the i'th centroid
                ind_followers = np.array([ind for ind, cent in enumerate(ret) if cent == i])
                # matrix of samples(real-values) following the i'th centroid
                followers = np.array([sample for ind, sample in enumerate(X) if (ind in ind_followers)])
                mean_vec = np.mean(followers, axis=0)  # average
                self.centroids[i] = mean_vec
            p = p + 1
            if not infinite_loop:
                if p >= max_iter:
                    break
            elif np.size(last_attributed_centroids) > 0:  # avoid comparaison on first iteration since it's empty
                # if no max_iter specified then run till getting no centroids change over 2-iterations
                cmp = np.equal(ret, last_attributed_centroids)  # compare attributed centroids one-by-one
                if np.product(cmp) == 1:  # perform and-logic
                    break
            last_attributed_centroids = ret
        self.aff_ = ret
        self.fitted = True

    def get_clusters(self):
        # return a list containing the input_data subdivided into clusters (specified in attr_vec)
        if not self.fitted:
            raise RuntimeError('Call fit over data before trying to get clusters!')
        clusts_ = []
        for i in range(self.K):
            X_ci = np.array([row for ind, row in enumerate(self.X) if self.aff_[ind] == i])
            clusts_.append(X_ci)
        return clusts_


def plot2d(X_input, format='rx'):
    X_f1 = X_input[:, 0]
    X_f2 = X_input[:, 1]
    plt.plot(X_f2, X_f1, format, color='#1d1d1d')


total_samples = 200
X = np.random.random_sample((total_samples, 2)) * 100

model = kMeans()
model.fit(X, 3, max_iter=-1)
clusters = model.get_clusters()

plot2d(clusters[0], format='r1')
plot2d(clusters[1], format='r.')
plot2d(clusters[2], format='r.')

plt.show()
