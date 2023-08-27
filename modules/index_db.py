from __future__ import annotations
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

logger = logging.getLogger(__name__)

BITS2DTYPE = {
    8: np.uint8,
}

class IndexDB:
    def __init__(self, d: int, m: int, nbits: int,
                 **estimator_kwargs: str | int) -> None:
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")
        if nbits not in BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nbits}")
        self.m = m
        self.k = 2 ** nbits
        self.d = d
        self.ds = d // m # the length of the segment
        self.estimators = [
            KMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(m)
        ]
        logger.info(f"Craeting following estimators: {self.estimators[0]!r}")
        self.is_trained = False
        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32
        self.codes: np.array | None = None

    def train(self, X: np.ndarray) -> None:
        """Train all KMeans estimators.
        Parameters
        ----------
        X
            Array of shape `(n, d)` and dtype `float32`.
        """
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed.")
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i*self.ds: (i+1)*self.ds]
            logger.info(f"Fitting KMeans for the {i}-th segment")
            estimator.fit(X_i)
        self.is_trained = True

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode original features into codes.
        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.
        Return
        ------
        result
            Array of shape `(n_queries, m)` of dtype `np.uint8`.
        """
        n = len(X)
        result = np.empty((n, self.m), dtype=self.dtype)
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i*self.ds: (i+1)*self.ds]
            result[:, i] = estimator.predict(X_i)

        return result

    def add(self, X: np.ndarray) -> None:
        """Add Vectors to the database (their encoded versions).
        Parameters
        ----------
        X
            Array of shape `(n_codes, d)` of dtype `float32`
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        self.codes = self.encode(X)

    def compute_asymmetric_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute asymmetric distances to all database codes.
        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.
        Return
        ------
        distances
            Array of shape `(n_queries, n_codes)` of dtype `np.float32`.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        if self.codes is None:
            raise ValueError("No codes detected. You need to run `add` first.")
        n_queries = len(X)
        n_codes = len(self.codes)
        distance_table = np.empty(
            (n_queries, self.m, self.k), dtype=self.dtype_orig
        )
        # for i in range(self.m):
        #     X_i = X[:, i*self.ds: (i+1)*self.ds] # (n_queries, ds)
        #     centers = self.estimators[i].cluster_centers_ # (k, ds)
        #     distance_table[:, i, :] = euclidean_distances(
        #         X_i, centers, squared=True
        #     )

        for i in range(self.m):
            X_i = X[:, i*self.ds: (i+1)*self.ds] # (n_queries, ds)
            centers = self.estimators[i].cluster_centers_ # (k, ds)
            distance_table[:, i, :] = cosine_similarity(
                X_i, centers
            )
        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)
        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]
        return distances

    def search(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Find k closest database codes to given queries.
        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`
        k
            The number of closest codes to look for.
        Return
        ------
        distances
            Array of shape `(n_queries, k)`
        indices
            Array of shape `(n_queries, k)`.
        """
        n_queries = len(X)
        distances_all = self.compute_asymmetric_distances(X)
        # indices = np.argsort(distances_all, axis=1)[:, :k]
        indices = np.argsort(-distances_all, axis=1)[:, :k]
        # indices = np.argsort(distances_all, axis=1)[::-1][:, :k]
        distances = np.empty((n_queries, k), dtype=np.float32)
        for i in range(n_queries):
            distances[i] = distances_all[i][indices[i]]
        return distances, indices


if __name__ == '__main__':
    d = embeds.shape[1]
    m = 8
    nbits = 8
    index = IndexDB(d, m, nbits, init='random', max_iter=10)
    logging.basicConfig(level=logging.INFO, force=True)