from testing_layer.configuration import Config, prepare_counted_values_output_dir
import json
import pandas as pd
from visualization_layer.plots import confussion
from visualization_layer.calculations import *
from testing_layer.workflows.augumentations import *
from os.path import isfile


def min_max_scaling(x: np.ndarray): return  (x - np.min(x)) / (np.max(x) - np.min(x))


def make_calculations(loaded_config: Config):
    """ Przeprowadź obliczenia odległości, i innych statystyk na podstawie konfiguracji. """
    train = pd.read_pickle("./cifar_10.pickle")
    cosine = CosineDistance()
    mahalanobis = MahalanobisDistance()
    euclidean = EuclidianDistance()
    k_nearest_neighbours = NearestNeightboursCount(100)

    train['features'] = train['features'].apply(min_max_scaling)  # data standarization
    # prepare train features for distance counting
    mahalanobis.fit(train)
    euclidean.fit(train)
    cosine.fit(train)
    k_nearest_neighbours.fit(train)
    for augumentation in loaded_config.augumentations:
        print("perform calculations for {}".format(augumentation.name))
        base_dir = "{}/dataframes".format(augumentation.template_path)
        base_output_dir = loaded_config.count_base_dir.joinpath(
            augumentation.template_path)
        iterator = augumentation.make_iterator()
        acc_wide = {k: [] for k in range(len(loaded_config.labels))}
        individual_calculations_dir = base_output_dir.joinpath("distances")
        print("Preparations for counting complete")
        for step in iterator:
            step = round(step, 2)
            print("Counting step: {}".format(step))
            file_path = base_dir + "/{}.pickle".format(step)
            if isfile(individual_calculations_dir.joinpath("all-distances-step-{}.pickle".format(step))):
                print("File detected, skipping")
                continue
            output_path_matrix = base_output_dir.joinpath("matrixes/{}".format(step))
            df = pd.read_pickle(file_path)
            df['features'] = df['features'].apply(min_max_scaling)
            y_pred = df['predicted_label'].values
            y_true = df['original_label'].values
            confussion.make_noise_wide_statistic(
                y_true, y_pred, range(len(loaded_config.labels)), acc_wide,
                filename=output_path_matrix)
            neighbours = k_nearest_neighbours.count_neighbours(df)
            dist_cosine = cosine.count_distance(df)
            dist_mahalanobis = mahalanobis.count_distance(df).T
            dist_euclidean = euclidean.count_distance(df)

            distances = {k: {
                "mahalanobis": dist_mahalanobis[k],
                "cosine": dist_cosine[k],
                "neighbours": neighbours[k],
                "euclidean": dist_euclidean[k],
                "original_label": df.iloc[k]['original_label'],
                "predicted_label": df.iloc[k]['predicted_label']
            } for k in range(len(loaded_config.dataset.targets))
            }
            pd.DataFrame.from_dict(distances, orient="index").to_pickle(
                individual_calculations_dir.joinpath("all-distances-step-{}.pickle".format(step)))
        with open(base_output_dir.joinpath("classes-accuracy.json"), "w+") as file:
            json.dump(acc_wide, file)


if __name__ == "__main__":
    with open("./config.json", "r") as file:
        obj = json.load(file)
    conf = Config(obj)
    prepare_counted_values_output_dir(conf)
    make_calculations(conf)
