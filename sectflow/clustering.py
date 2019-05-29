import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class TrajClust:
    def __init__(
        self,
        features,
        min_cluster_size_ratio=0.02,
        sub_min_cluster_size_ratio=0.02,
        eps=0.4,
        sub_eps=0.5,
        min_samples_ratio=0.01,
        flow_merge_dist=15,
        outlier_merge_dist=30,
    ):

        self.features = features
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.sub_min_cluster_size_ratio = sub_min_cluster_size_ratio
        self.eps = eps
        self.sub_eps = sub_eps
        self.min_samples_ratio = min_samples_ratio
        self.flow_merge_dist = flow_merge_dist
        self.outlier_merge_dist = outlier_merge_dist

    def fit(self, X):
        self.interp_points_num = X.shape[1]

        # 1st level clustering
        logging.info("Clustering...")
        min_cluster_size = int(X.shape[0] * self.min_cluster_size_ratio)
        labels, centroids, b, bools = self._main_clustering(
            X, min_cluster_size, self.eps
        )
        labels, centroids = self._check_unclustered_traj(X, labels, centroids)
        self._labels_statistics(labels)

        # recursive clustering
        logging.info("Sub-Clustering...")
        sub_min_cluster_size = int(X.shape[0] * self.sub_min_cluster_size_ratio)
        b = True
        while b:
            labels, centroids, b, bools = self._main_clustering(
                X, sub_min_cluster_size, self.sub_eps, labels, centroids, bools
            )
        labels, centroids = self._check_unclustered_traj(X, labels, centroids)
        self._labels_statistics(labels)

        # cluster merging
        logging.info("Cluster merging...")
        labels, centroids = self._fusion_loop(centroids, X, labels)
        traj_ids = range(len(X))
        for k, (id, l) in enumerate(zip(traj_ids, labels)):
            if l != -1:
                new_id = self._best_hosting_cluster(X[id], centroids, labels)
                if new_id != None:
                    labels[k] = new_id

        ratio_dict = {
            a: sum([l == a for l in labels]) / len(labels)
            for a in np.unique(labels)
        }
        b = False
        for id in ratio_dict:
            r = ratio_dict[id]
            if r < self.min_cluster_size_ratio:
                b = True
                for k, l in enumerate(labels):
                    if l == id:
                        labels[k] = -1
        if b:
            labels, centroids = self._update_cluster_ids(labels, X)
            labels, centroids = self._check_unclustered_traj(
                X, labels, centroids
            )
        self._labels_statistics(labels)

        self.labels_ = labels

    def _main_clustering(
        self, sv, min_cluster_size, eps, labels=None, centroids=None, bools=None
    ):
        B = False
        if labels is None:
            labels, centroids = self._clustering(sv, min_cluster_size, eps)
            sub_sv = sv[labels == -1]
            sub_labels, sub_centroids = self._clustering(
                sub_sv, min_cluster_size, eps
            )
            labels, b = self._merge_labels(labels, -1, sv, sub_labels, sub_sv)
            B = True
            bools = [True for a in range(max(labels) + 1)] + [True]
        else:
            nb_cluster = max(labels) + 1
            for cluster_id in range(nb_cluster):
                if bools[cluster_id]:
                    sub_sv = sv[labels == cluster_id]
                    sub_labels, sub_centroids = self._clustering(
                        sub_sv, min_cluster_size, eps
                    )
                    labels, b = self._merge_labels(
                        labels, cluster_id, sv, sub_labels, sub_sv
                    )
                    B += b
                    bools[cluster_id] = b
            if bools[-1]:
                sub_sv = sv[labels == -1]
                sub_labels, sub_centroids = self._clustering(
                    sub_sv, min_cluster_size, eps
                )
                labels, b = self._merge_labels(
                    labels, -1, sv, sub_labels, sub_sv
                )
                B += b
                bools[-1] = b

        labels, centroids = self._update_cluster_ids(labels, sv)

        bools = (
            bools[:-1]
            + [True for a in range(max(labels) + 1 - (len(bools) - 1))]
            + [bools[-1]]
        )
        return (labels, centroids, B, bools)

    def _clustering(self, sv, min_cluster_size, eps):
        labels, min_cluster_size = self._entry_exit_clustering3D(
            sv, min_cluster_size=min_cluster_size, eps=eps
        )
        centroids = self._create_centroids(sv, labels)
        return (labels, centroids)

    def _idx_feature(self, feature):
        for i, f in enumerate(self.features):
            if f == feature:
                return i
        return None

    def _entry_exit_clustering3D(self, matrix, min_cluster_size, eps):
        nb = len(matrix)
        min_samples = max(int(nb * self.min_samples_ratio), 3)
        features = []
        if "latitude" in self.features and "longitude" in self.features:
            features.append(self._idx_feature("latitude"))
            features.append(self._idx_feature("longitude"))
        else:
            features.append(self._idx_feature("x"))
            features.append(self._idx_feature("y"))
        if "log_altitude" in self.features:
            features.append(self._idx_feature("log_altitude"))
        elif "altitude" in self.features:
            features.append(self._idx_feature("altitude"))

        entry = matrix[:, : len(self.features)]
        exit = matrix[:, -len(self.features) :]
        entry, exit = entry[:, features], exit[:, features]

        matrix = np.hstack([entry, exit])
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            StandardScaler().fit_transform(matrix)
        )

        nb_cluster = max(labels) + 1
        L = [0 for k in range(nb_cluster)]
        for l in labels:
            if l != -1:
                L[l] += 1

        for k, count in enumerate(L):
            if count < min_cluster_size:
                for i, l in enumerate(labels):
                    if l == k:
                        labels[i] = -1

        self.set_new_ids(labels)

        return (labels, min_cluster_size)

    def _merge_labels(self, labels, cluster_id, sv, sub_labels, sub_sv):

        if max(sub_labels) >= 0:
            nb_cluster = max(labels)
            sub_traj_ids = np.where(labels == cluster_id)[0]
            for k, i in zip(sub_labels, sub_traj_ids):
                if cluster_id != -1:
                    if k != -1:
                        if k == 0:
                            labels[i] = cluster_id
                        else:
                            labels[i] = nb_cluster + k
                    else:
                        labels[i] = -1
                else:
                    if k != -1:
                        labels[i] = nb_cluster + 1 + k
                    else:
                        labels[i] = -1

            if cluster_id == -1:
                return (labels, True)
            elif max(sub_labels) == 0:
                return (labels, False)
            else:
                return (labels, True)

        else:
            return (labels, False)

    def _check_unclustered_traj(self, sv, labels, centroids):
        for _, idx in enumerate(np.where(labels == -1)[0]):
            host_id = self._best_hosting_cluster(sv[idx], centroids, labels)
            if isinstance(host_id, int):
                labels[idx] = host_id
        return (labels, centroids)

    def _update_cluster_ids(self, labels, sv):
        self.set_new_ids(labels)
        centroids = self._create_centroids(sv, labels)
        return labels, centroids

    def _labels_statistics(self, labels):
        nb_cluster = max(labels) + 1
        nb = len(labels)
        for k in range(nb_cluster):
            count = len(labels[labels == k])
            logging.info(
                "Cluster {}: {} trajectories = {}% of the traffic".format(
                    k, count, round(count / nb * 100, 1)
                )
            )
        count = len(labels[labels == -1])
        logging.info(
            "Nb singleton: {} trajectories = {}% of the traffic".format(
                count, round(count / nb * 100, 1)
            )
        )

    def _fusion_loop(self, centroids, sv, labels):
        B = True
        count = -1
        while B:
            count += 1
            B = False
            nb_cluster = max(labels) + 1
            d = {}
            for i in range(nb_cluster):
                j = self._best_hosting_cluster(i, centroids, labels)
                if isinstance(j, int) and j < i:
                    B = True
                    d[i] = j
            for i in sorted(d, reverse=True):
                labels, b = self._fusion(i, d[i], labels, centroids)
                if b:
                    labels, centroids = self._update_cluster_ids(labels, sv)

        return (labels, centroids)

    def _create_centroids(self, trajectories, labels):
        centroids = [
            np.mean(trajectories[labels == l], axis=0)
            for l in np.unique(labels)
            if l != -1
        ]
        if centroids:
            centroids = np.vstack(centroids)
        return centroids

    def _get_features(self, item):
        first = item[0 : len(self.features)]
        last = item[-len(self.features) :]
        iy = self._idx_feature("y")
        ix = self._idx_feature("x")
        ia = self._idx_feature("altitude")
        return (
            first[iy],
            first[ix],
            last[iy],
            last[ix],
            0 if ia is None else first[ia],
            0 if ia is None else last[ia],
        )

    def _best_hosting_cluster(self, item, centroids, labels):
        m = 100000000
        host_clust_id = None
        if isinstance(item, int):
            U0, UN, Ualt, U_evol = self.get_vect(centroids[item])
            for k in range(len(centroids)):
                if k != item:
                    dist = self.get_dist(
                        centroids[k], U0, UN, Ualt, U_evol, self.flow_merge_dist
                    )
                    if dist is not None and dist < m:
                        host_clust_id = k
                        m = dist
        else:
            U0, UN, Ualt, U_evol = self.get_vect(item)
            for k in range(len(centroids)):
                dist = self.get_dist(
                    centroids[k], U0, UN, Ualt, U_evol, self.outlier_merge_dist
                )
                if dist is not None and dist < m:
                    host_clust_id = k
                    m = dist

        return host_clust_id

    def get_dist(self, centroid, U0, UN, Ualt, U_evol, merge_dist):
        V0, VN, Valt, V_evol = self.get_vect(centroid)
        dist0 = np.sqrt(sum((U0 - V0) ** 2)) / 1852
        distN = np.sqrt(sum((UN - VN) ** 2)) / 1852
        dh = abs(Ualt - Valt)
        dist = None
        if (
            dist0 < merge_dist and distN < merge_dist and V_evol == U_evol
        ):  # dh[0] < 7000 and dh[1]<7000:
            dist = np.sqrt((dist0 * 1852) ** 2 + dh[0] ** 2) + np.sqrt(
                (distN * 1852) ** 2 + dh[1] ** 2
            )

        return dist

    def _check_cluster_proximity(self, cluster_id, host_cluster_id, centroids):
        U0, UN, _, U_evol = self.get_vect(centroids[cluster_id])
        V0, VN, _, V_evol = self.get_vect(centroids[host_cluster_id])

        dist0 = np.sqrt(sum((U0 - V0) ** 2)) / 1852
        distN = np.sqrt(sum((UN - VN) ** 2)) / 1852

        return (
            dist0 < self.flow_merge_dist
            and distN < self.flow_merge_dist
            and U_evol == V_evol
        )

    def get_vect(self, item):
        y0, x0, yN, xN, a0, aN = self._get_features(item)
        U0 = np.array([x0, y0])
        UN = np.array([xN, yN])
        Ualt = np.array([a0, aN])
        U_evol = abs(Ualt[0] - Ualt[1]) > 4000
        if U_evol and Ualt[0] - Ualt[1] < 0:
            U_evol *= -1
        return U0, UN, Ualt, U_evol

    def _fusion(self, cluster_id, host_cluster_id, labels, centroids):
        b = False
        if self._check_cluster_proximity(
            cluster_id, host_cluster_id, centroids
        ):
            b = True
            logging.info(f"fusion {cluster_id} -> {host_cluster_id}")
            for k, l in enumerate(labels):
                if l == cluster_id:
                    labels[k] = host_cluster_id
            self.set_new_ids(labels)
        return (labels, b)

    def set_new_ids(self, labels):
        sorted_cluster_ids = sorted(list(pd.Series(labels).unique()))
        new_cluster_ids = [k for k in range(-1, len(sorted_cluster_ids) - 1)]
        for k, l in enumerate(labels):
            if l != -1:
                labels[k] = new_cluster_ids[sorted_cluster_ids.index(l)]
