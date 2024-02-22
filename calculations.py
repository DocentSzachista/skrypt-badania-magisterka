from testing_layer.configuration import Config, prepare_counted_values_output_dir
import json
import pandas as pd
from visualization_layer.plots import confussion
from visualization_layer.calculations import *
from testing_layer.workflows.augumentations import *
from os.path import isfile
from testing_layer.workflows.enums import SupportedModels
from testing_layer.datasets import MixupDataset, ImageNetKaggle
from multiprocessing import Pool
from functools import partial
from testing_layer.model_loading import supported_weights
import logging
logging.basicConfig(level=logging.DEBUG, format='%(processName)s: %(message)s')
from sklearn.metrics import confusion_matrix


def softmax(vector: np.ndarray):
    e = np.exp(vector)
    return e / e.sum()


def min_max_scaling(x: np.ndarray): return  (x - np.min(x)) / (np.max(x) - np.min(x))


def calculate_confussion_matrix(dataframe: pd.DataFrame, output_path: pathlib.Path, step: int):
    output_path = output_path.joinpath("matrixes/")
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path.joinpath("matrix-step-{}.npy".format(step))
    if isfile(save_path):
        print("File detected, skipping")
        return
    global labels
    y_pred = dataframe['predicted_label'].values.astype(np.int16)
    y_true = dataframe['original_label'].values.astype(np.int16)
    cfm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)

    np.save(save_path, cfm)


def count_softmax(dataframe: pd.DataFrame, output_path: pathlib.Path, step: int):
    logging.info("Counting step: {}".format(step))

    output_path = output_path.joinpath("softmax")
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path.joinpath("step-{}.pickle".format(step))
    if isfile(save_path):
        print("Is already counted, skipping")
        return

    ids = dataframe.id.values
    original_labels = dataframe.original_label.values
    recognized_labels = dataframe.predicted_label.values
    logits =  np.stack(dataframe['classifier'].values)
    softmaxed = np.apply_along_axis(softmax, 1, logits)
    to_save = pd.DataFrame(
        {
            "id": ids,
            "original_label": original_labels,
            "predicted_labels": recognized_labels,
            "softmaxed_values": list(softmaxed)
        }
    )
    output_path = output_path.joinpath("softmax")
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info("Saving step {}".format(step))
    to_save.to_pickle(save_path)


def count_distance(file_path: str, step: int, output_path: pathlib.Path,
                   distance_method: MahalanobisDistance | CosineDistance | EuclidianDistance,
                   targets: list
                   ):
    try:
        output_path = output_path.joinpath("{}/".format(distance_method.name))
        output_path.mkdir(parents=False, exist_ok=True)
        output_path = output_path.joinpath("distance-step-{}.pickle".format( step))
        # if isfile(output_path):
        #     logging.info("Found file. Skipping step {}".format(step))
        #     return
        logging.info("Counting step: {}".format(step))
        dataframe = pd.read_pickle(file_path)
        logging.info("Scaling dataframe {}".format(step))
        dataframe['features'] = dataframe.features.apply(min_max_scaling)
        logging.info("Counting distance {}".format(step))
        counted_distance  = distance_method.count_distance(dataframe)
        if isinstance(distance_method, MahalanobisDistance):
            counted_distance = counted_distance.T
        logging.info("Creating dataframe with results {}".format(step))
        distances = {k: {
            distance_method.name: counted_distance[k],
            "original_label": dataframe.iloc[k]['original_label'],
            "predicted_label": dataframe.iloc[k]['predicted_label']
        } for k in range(50000)
        }
        pd.DataFrame.from_dict(distances, orient="index").to_pickle(output_path)
        logging.info("Saving step {}".format(step))
    except Exception as e:
        logging.error("Got unexpected error in step {} \n Stacktrace {}".format(step, e))




def perform_counting_per_method(config: Config, should_use_multiprocessing: bool | None = False):
    global labels
    labels = list(range(0, len(config.labels)))
    euclidean = EuclidianDistance()
    cosine  = CosineDistance()
    mahalanobis = MahalanobisDistance()

    methods_to_count = {
        # "softmax": count_softmax,
        # "matrixes": calculate_confussion_matrix,
        # "euclidean": euclidean,
        "cosine": cosine,
        # "mahalanobis": mahalanobis
    }


    for augumentation in config.augumentations:
        iterator = augumentation.make_iterator()
        base_dir = "{}/dataframes".format(augumentation.template_path)
        base_output_dir = config.count_base_dir.joinpath(
            augumentation.template_path)
        file_path = base_dir + "/{}.pickle".format(0)
        train = pd.read_pickle(file_path)
        train['features'] = train.features.apply(min_max_scaling)
        for key, method in methods_to_count.items():
            print("Counting chosen method: {}".format(key))
            if key == "euclidean":
                euclidean.fit(train)
            elif key == "cosine":
                cosine.fit(train)
            elif key == "mahalanobis":
                mahalanobis.fit(train)
            if should_use_multiprocessing:
                with Pool(processes=config.num_workers) as pool:
                    pool.starmap(
                        count_distance, [(base_dir + "/{}.pickle".format(step), step, base_output_dir, method, labels) for step in iterator ]
                    )
            else:
                for step in iterator:
                    step = round(step, 2)
                    print("Count step: {}".format(step))
                    file_path = base_dir + "/{}.pickle".format(step)
                    if isinstance(method, Metrics):
                        count_distance(
                            file_path, step, base_output_dir, method, labels
                        )
                    else:
                        dataframe = pd.read_pickle(file_path)
                        dataframe['features'] = dataframe.features.apply(min_max_scaling)
                        method(dataframe, base_output_dir, step)
                print("Saving step")


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
    with open("./config-imagenet.json", "r") as file:
        obj = json.load(file)
        models = [SupportedModels(model) for model in obj.get("models")]

        dataset = ImageNetKaggle(root=obj['dataset_path'], split="val", transform=None)

    for tested_model in models:
        obj['model'] = tested_model.value
        conf = Config(obj)
        conf.dataset = dataset
        # weights = supported_weights[tested_model]
        # conf.augumentations[0].se
        print("Count model: {}".format(tested_model.value))
        prepare_counted_values_output_dir(conf)
        perform_counting_per_method(conf, True if conf.num_workers > 1 else False )
        # count_softmax_values(conf)
        # make_calculations(conf)