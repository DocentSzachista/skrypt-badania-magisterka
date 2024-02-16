import numpy as np
import pandas as pd

from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


class Metrics:
    """Abstract class that for counting metrics"""
    name = "abstract"
    y_lim = (0, 10)

    def __init__(self) -> None:
        self.dist = {}
        self.classes = {}

    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        raise NotImplementedError("This is abstract method, its not gonna be implemented")


class EuclidianDistance(Metrics):
    """Class that implements counting euclidan distance."""

    name = "Euclidian"
    y_lim = (0, 10)

    def fit(self, df: pd.DataFrame):
        self.classes = np.sort(df.original_label.unique())
        for index in self.classes:
            x = np.stack(df[df['original_label'] == index]['features'].to_numpy())
            # print(x.shape)
            self.dist[index] = np.stack(x.mean(axis=0))
            # print(self.dist[index].shape)

    def count_distance(self, df: pd.DataFrame):
        """Licz średnią odległość euklidesową od zbioru punktów dla zbioru x"""
        x = df['features'].to_numpy()
        x = np.stack(x).squeeze()  # .tolist()
        res = []
        for row in x:
            row_dist = []
            for class_ in self.classes:
                instance = np.linalg.norm(self.dist[class_] - row)
                row_dist.append(instance)
            res.append(row_dist)
        res = np.stack(res, axis=0)
        print(res.shape)
        print(res[0])
        return res


class CosineDistance(Metrics):
    """Class that implements counting cosine distance."""

    name = "Cosine"
    y_lim = (0, 1)

    def fit(self, df: pd.DataFrame):
        self.classes = np.sort(df.original_label.unique())
        for index in self.classes:
            x = df[df['original_label'] == index]['features'].to_numpy()
            self.dist[index] = x.mean(axis=0)

    def count_distance(self, df: pd.DataFrame):
        """ Oblicz średnią odległość cosinusową od zestawu punktów"""
        x = df['features'].to_numpy()
        x = np.stack(x).squeeze()
        res = []
        for row in x:
            row_dist = []
            for class_ in self.classes:
                instance = cosine_distances(self.dist[class_], [row])
                row_dist.append(instance)
            res.append(row_dist)
        res = np.stack(res, axis=0)
        print(res.shape)
        print(res[0])
        return res


class MahalanobisDistance:
    """Class to count mahalanobis distance."""

    name = "Mahalanobis"

    def __init__(self) -> None:
        self.dist = {}
        self.classes = {}

    def fit(self, df: pd.DataFrame):
        self.classes = np.sort(df.original_label.unique())
        # print(self.classes)
        for index in self.classes:
            x = df[df['original_label'] == index]['features'].tolist()
            self.dist[index] = EmpiricalCovariance().fit(x)

    def count_distance(self, df: pd.DataFrame):
        x = df['features'].to_numpy()

        x = np.stack(x).squeeze().tolist()
        res = np.stack([np.sqrt(self.dist[index].mahalanobis(x)) for index in self.classes], axis=0)
        return res


class NearestNeightboursCount:

    name = "nearestneightbours"

    def __init__(self, n_neighbors: int) -> None:
        self.classes = {}
        self.classes_indicises = {}
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, df: pd.DataFrame):
        self.classes = np.sort(df.original_label.unique())

        for index in self.classes:
            x = df[df['original_label'] == index].index
            self.classes_indicises[index] = x
        self.knn.fit(
            X=df['features'].to_list()
        )

    def count_neighbours(self, df: pd.DataFrame):
        """Podlicz liczbę sąsiadów a następnie podlicz z jakiej klas są."""
        x = df['features'].to_numpy()
        x = np.stack(x).squeeze()
        neighbours = self.knn.kneighbors(x, return_distance=False)
        to_return_list = []
        for neightbor in neighbours:
            to_return = {
                k: len(list(set(neightbor) & set(self.classes_indicises[k]))) for k in self.classes
            }
            to_return_list.append(to_return)
        return to_return_list
