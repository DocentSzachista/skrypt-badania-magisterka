import numpy as np
import pandas as pd
from numpy.linalg import norm

from sklearn.covariance import EmpiricalCovariance

class Distance:
    """Abstract class that for measuring distance"""
    name = "abstract"
    y_lim = (0, 10)

    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        raise NotImplementedError("This is abstract method, its not gonna be implemented")


class EuclidianDistance(Distance):
    """Class that implements counting euclidan distance."""

    name = "Euclidian"
    y_lim = (0, 10)

    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        """Count euclidean distance.

            :param: origin
            Starting point (original image's features, or final augumentation version)
            :param: target
            Ending point   (point in the space between origin and final version)
        """
        return norm(features_origin - features_finish)


class CosineDistance(Distance):
    """Class that implements counting cosine distance."""

    name = "Cosine"
    y_lim = (0, 1)

    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        """Count cosine distance
            :param: origin
            Starting point (original image's features, or final augumentation version)
            :param: target
            Ending point   (point in the space between origin and final version)
        """
        return 1 - float(np.dot(
            features_origin, features_finish.T
        )/(
            norm(features_origin)*norm(features_finish)))



class MahalanobisDistance:
    """Class to count mahalanobis distance."""

    name = "Mahalanobis"

    def __init__(self) -> None:
        self.dist = {}
        self.classes = {}

    def fit(self, df: pd.DataFrame):
        self.classes = df.original_label.unique()

        for index in self.classes:
            x = df[df['original_label'] == index]['features'].tolist()
            self.dist[index] = EmpiricalCovariance().fit(x)

    def count_distance(self, df: pd.DataFrame):
        x = df['features'].to_numpy()

        x = np.stack(x).squeeze().tolist()
        res = np.stack([np.sqrt(self.dist[index].mahalanobis(x)) for index in self.classes], axis=0)
        # res = res.min(axis=0)
        return res
