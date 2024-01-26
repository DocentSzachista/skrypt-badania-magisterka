from testing_layer.configuration import Config, prepare_counted_values_output_dir
import json
import pandas as pd
from visualization_layer.plots import confussion
from visualization_layer.calculations import *
from testing_layer.workflows.augumentations import *
from os.path import isfile

def softmax(vector: np.ndarray):
    e = np.exp(vector)
    return e / e.sum()


def min_max_scaling(x: np.ndarray): return  (x - np.min(x)) / (np.max(x) - np.min(x))


def make_calculations(loaded_config: Config):
    """ Przeprowadź obliczenia odległości, i innych statystyk na podstawie konfiguracji."""
    print(loaded_config.chosen_train_set)
    train = pd.read_pickle(loaded_config.chosen_train_set)
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


def count_softmax_values(config: Config):
    for augumentation in config.augumentations:
        base_dir = "{}/dataframes".format(augumentation.template_path)
        base_output_dir = config.count_base_dir.joinpath(
            augumentation.template_path)
        iterator = augumentation.make_iterator()
        individual_calculations_dir = base_output_dir.joinpath("distances")
        for step in iterator:
            step = round(step, 2)
            print("Counting step: {}".format(step))
            file_path = base_dir + "/{}.pickle".format(step)
            if isfile(individual_calculations_dir.joinpath("softmax-step-{}.pickle".format(step))):
                print("File detected, skipping")
                continue
            df = pd.read_pickle(file_path)
            ids = df.id.values
            original_labels = df.original_label.values
            recognized_labels = df.predicted_label.values
            logits =  np.stack(df['classifier'].values)
            print(logits.shape)
            softmaxed = np.apply_along_axis(softmax, 1, logits)
            to_save = pd.DataFrame(
                {"id": ids,
                 "original_label": original_labels,
                 "predicted_labels": recognized_labels,
                 "softmaxed_values": list(softmaxed)
                }
            )
            to_save.to_pickle(individual_calculations_dir.joinpath("softmax-step-{}.pickle".format(step)))


def count_best_100_neighbours(config: Config):
    k_nearest = NearestNeightboursCount(n_neighbors=config.number_neighbours)
    for augumentation in config.augumentations:
        base_dir = "{}/dataframes".format(augumentation.template_path)
        base_output_dir = config.count_base_dir.joinpath(
            augumentation.template_path)
        iterator = augumentation.make_iterator()
        individual_calculations_dir = base_output_dir.joinpath("distances")
        for step in iterator:
            ids = []
            original_labels = []
            recognized_labels = []
            step = round(step, 2)
            print("Counting step: {}".format(step))
            file_path = base_dir + "/{}.pickle".format(step)
            if isfile(individual_calculations_dir.joinpath("nearest-neightbour-step-{}.pickle".format(step))):
                print("File detected, skipping")
                continue
            df = pd.read_pickle(file_path)
            features = df.features.values
            if step == 0:
                k_nearest.fit(features)
            else:
                chosen = df.groupby('original_label').apply(lambda x: x.sample(n=100), random_state=0)
                for state, frame in chosen:
                    nearest_neightbours = k_nearest.count_neighbours(frame)

            # to_save = pd.DataFrame(
            #     {"id": ids,
            #      "original_label": original_labels,
            #      "predicted_labels": recognized_labels,
            #      "softmaxed_values": list(softmaxed)
            #     }
            # )
            # to_save.to_pickle(individual_calculations_dir.joinpath("softmax-step-{}.pickle".format(step)))



if __name__ == "__main__":
    filenames = ["config-noise.json", "config-noise-shuffle.json", "config-mixup.json", "config-mixup-shuffle.json"]
    for filename in filenames:
        with open("./{}".format(filename), "r") as file:
            obj = json.load(file)
        conf = Config(obj)
        prepare_counted_values_output_dir(conf)
        count_softmax_values(conf)
    # make_calculations(conf)
    # import shutil
    # shutil.make_archive("counted_outputs", "zip", conf.count_base_dir)