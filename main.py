import numpy as np
from matplotlib import pyplot as plt


class kMeans:
    def __init__(self):
        self.X = None  # data
        self.n_clusters = 2  # number of clusters
        self.tol = 1e-3  # tolerance
        self.fitted = False  # fit called ?
        self.centroids = None  # list of centroids coords
        self.aff_ = None  # array of affected centroids' indices

    def fit(self, X, n_clusters=2, tol=1e-3, verbose=0):
        # will return: [final centroids], array of (n_samples, ) containing attributed centroid's index
        self.X = X
        self.n_clusters = n_clusters
        self.tol = tol
        self.fitted = True
        # let's begin with randomly init centroids
        cent_coord_list = []  # a list of coords (tuple)
        cent_index_list = []  # cent indices for unique selection
        (m, n) = X.shape
        for c in range(n_clusters):
            rd = np.random.randint(0, m, dtype=np.int32)
            while rd in cent_index_list:
                rd = np.random.randint(0, m, dtype=np.int32)
            c_coord = X[rd, :]
            cent_coord_list.append(c_coord)
            cent_index_list.append(rd)
        self.centroids = np.array(cent_coord_list)
        self.aff_ = np.zeros((m,), dtype=np.int32)
        ex_variation = np.inf  # init
        cur_variation = 0  # init
        iter_ = 0  # iteration counter
        while True:
            for i in range(m):
                point = X[i, :]
                attributed_cent = 0
                attributed_norm = np.linalg.norm(point - self.centroids[0, :])
                for j in np.arange(n_clusters - 1) + 1:
                    cent = self.centroids[j, :]
                    norm = np.linalg.norm(point - cent)  # distance
                    if norm < attributed_norm:
                        attributed_norm = norm
                        attributed_cent = j
                self.aff_[i] = attributed_cent
            # now edit each centroids to the mean of its followers
            for i in range(n_clusters):
                # array of samples-indexes following the i'th centroid
                ind_followers = np.array([ind for ind, cent in enumerate(self.aff_) if cent == i])
                # matrix of samples(real-values) following the i'th centroid
                followers = np.array([sample for ind, sample in enumerate(X) if (ind in ind_followers)])
                mean_vec = np.mean(followers, axis=0)  # average
                self.centroids[i] = mean_vec
            # compute variation
            cur_variation = self.compute_variation()
            if verbose:
                print('iteration ', iter_, ' current_variation= ', cur_variation, ' ex_variation= ', ex_variation)
            if cur_variation <= ex_variation <= cur_variation + self.tol:
                break
            ex_variation = cur_variation
            iter_ += 1
        return self.centroids, self.aff_

    def get_clusters(self):
        # return a list containing the input_data subdivided into clusters (specified in attr_vec)
        if not self.fitted:
            raise RuntimeError('Call fit over data before trying to get clusters!')
        clusts_ = []
        for i in range(self.n_clusters):
            X_ci = np.array([row for ind, row in enumerate(self.X) if self.aff_[ind] == i])
            clusts_.append(X_ci)
        return clusts_

    def compute_cluster_variation(self, target_cluster):
        # return within cluster variation of the target_cluster
        if not self.fitted:
            raise RuntimeError('Call fit over data before trying to compute variation!')
        cent = self.centroids[target_cluster]  # coord of target cluster
        X_cent = np.array([row for ind, row in enumerate(self.X) if self.aff_[ind] == target_cluster])
        (n_samples, dummy) = X_cent.shape

        # compute the average squared euclidean distance sample <-> centroid
        return np.sum(np.sum((X_cent - cent) ** 2, axis=1)) / n_samples

    def compute_variation(self):
        # return average training variation of the target_cluster
        if not self.fitted:
            raise RuntimeError('Call fit over data before trying to compute error!')
        cumulative_variation = 0
        for i in range(self.n_clusters):
            cumulative_variation += self.compute_cluster_variation(target_cluster=i)
        return cumulative_variation


def plot2d(X_input, format='rx'):
    X_f1 = X_input[:, 0]
    X_f2 = X_input[:, 1]
    plt.plot(X_f2, X_f1, format, color='#1d1d1d')


total_samples = 200
X = np.random.random_sample((total_samples, 2)) * 100

model = kMeans()
model.fit(X, 3, tol=0, verbose=1)
clusters = model.get_clusters()

print('variation = ', model.compute_variation())

plot2d(clusters[0], format='rs')
plot2d(clusters[1], format='r.')
plot2d(clusters[2], format='r^')

plt.show()
