import numpy as np


class KMeans:
    def __init__(self):
        self.X = None  # data
        self.n_clusters = 2  # number of clusters
        self.tol = 1e-3  # variation tolerance
        self.fitted = False  # fit called ?
        self.centroids = None  # list of centroids coords
        self.aff_ = None  # array of affected centroids' indices
        self.var_ = np.inf  # variation

    def fit(self, X, n_clusters=2, tol=1e-3, verbose=0):
        # will return: self (to directly get var_, call get_clusters if needed)
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
        self.var_ = cur_variation
        return self

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

    def compute_silhouette(self):
        # return a list of n_clusters np.arrays
        # each array contains the silhouette coefficients of each simple the cluster holds
        if not self.fitted:
            raise RuntimeError('Call fit over data before trying to evaluate the model!')
        silhouettes = []
        clusters = self.get_clusters()
        for i in range(self.n_clusters):
            centroid = self.centroids[i]
            # first let's get the index of the closest centroid
            closest_centroid_index = -1
            distance = np.inf
            for j in range(self.n_clusters):
                if j == i:  # avoid same centroid
                    continue
                other_centroid = self.centroids[j]
                d = np.linalg.norm(centroid - other_centroid)
                if d < distance:
                    distance = d
                    closest_centroid_index = j
            subset = clusters[i]
            closest_subset = clusters[closest_centroid_index]
            (m, _) = subset.shape  # cluster's subset samples
            clust_sil = np.zeros((m,))
            for j in range(m):
                a = np.sum(
                    np.sqrt(np.sum((subset - subset[j]) ** 2, axis=1))) / m  # distance from samples within cluster
                b = np.sum(np.sqrt(np.sum((closest_subset - subset[j]) ** 2,
                                          axis=1))) / m  # distance from samples within the closest cluster
                clust_sil[j] = (b - a) / max(a, b)
            silhouettes.append(clust_sil)
        return silhouettes
