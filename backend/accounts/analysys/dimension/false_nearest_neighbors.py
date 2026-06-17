import numpy as np
from sklearn.neighbors import NearestNeighbors


def false_nearest_neighbors(x, max_dim=10, tau=10, Rt=2.0):

    def embed_time_series(x, m, tau):
        N = len(x) - (m - 1) * tau
        return np.array([x[i:i + m * tau:tau] for i in range(N)])

    N = len(x)
    fnn_percent = []
    for m in range(1, max_dim + 1):
        embedded = embed_time_series(x, m, tau)
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded)
        distances, indices = nbrs.kneighbors(embedded)
        false_count, valid = 0, 0
        for i in range(len(indices)):
            j = indices[i][1]
            if (i + 1 < len(x)) and (j + 1 < len(x)):
                num = abs(x[i + 1] - x[j + 1])
                denom = distances[i][1] if distances[i][1] > 0 else 1e-10
                Ri = num / denom
                if Ri > Rt:
                    false_count += 1
                valid += 1
        fnn_percent.append(false_count / valid * 100)
    return fnn_percent


def find_optimal_dim(fnn_ratios, threshold=10):
    for m, ratio in enumerate(fnn_ratios, start=1):
        if ratio < threshold:
            return m
    return len(fnn_ratios)
