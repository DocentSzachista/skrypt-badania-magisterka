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
    to_save.to_pickle(save_path)


def count_distance(dataframe: pd.DataFrame, step: int, output_path: pathlib.Path, 
                   distance_method: MahalanobisDistance | CosineDistance | EuclidianDistance,
                   targets: list
                   ): 
    counted_distance  = distance_method.count_distance(dataframe)
    if isinstance(distance_method, MahalanobisDistance):
        counted_distance = counted_distance.T
    distances = {k: {
        distance_method.name: counted_distance[k],
        "original_label": dataframe.iloc[k]['original_label'],
        "predicted_label": dataframe.iloc[k]['predicted_label']
    } for k in range(len(targets))
    }
    pd.DataFrame.from_dict(distances, orient="index").to_pickle(
        output_path.joinpath("{}/distance-step-{}.pickle".format(distance_method.name, step)))




def perform_counting_per_method(config: Config, should_use_multiprocessing: bool | None = False):
    global labels
    labels = list(range(0, len(config.labels)))
    euclidean = EuclidianDistance()
    cosine  = CosineDistance()
    mahalanobis = MahalanobisDistance()

    methods_to_count = {
        "softmax": count_softmax,
        "matrixes": calculate_confussion_matrix,
        # "euclidean": euclidean,
        # "cosine": cosine,
        # "mahalanobis": mahalanobis
    }


    for augumentation in config.augumentations:
        iterator = augumentation.make_iterator()
        base_dir = "{}/dataframes".format(augumentation.template_path)
        base_output_dir = config.count_base_dir.joinpath(
            augumentation.template_path)
        file_path = base_dir + "/{}.pickle".format(0)
        train = pd.read_pickle(file_path)
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
                    pool.map(
                        iterator
                    )
            else:
                for step in iterator:
                    step = round(step, 2)
                    print("Count step: {}".format(step))
                    file_path = base_dir + "/{}.pickle".format(step)
                    dataframe = pd.read_pickle(file_path)
                    if isinstance(method, Metrics):
                        dataframe['features'] = dataframe.features.apply(min_max_scaling)
                        count_distance(
                            dataframe, step, base_output_dir, method, config.labels
                        )
                    else:
                        method(dataframe, base_output_dir, step)
                print("Saving step")


            # dataframe['features'] = dataframe.features.apply(min_max_scaling)





# def make_calculations(loaded_config: Config):
#     """ Przeprowadź obliczenia odległości, i innych statystyk na podstawie konfiguracji."""
#     # print(loaded_config.chosen_train_set)
#     # train = pd.read_pickle(loaded_config.chosen_train_set)
#     cosine = CosineDistance()
#     mahalanobis = MahalanobisDistance()
#     euclidean = EuclidianDistance()
#     # k_nearest_neighbours = NearestNeightboursCount(100)

#     # prepare train features for distance counting
    
#     for augumentation in loaded_config.augumentations:
#         print("perform calculations for {}".format(augumentation.name))
#         base_dir = "{}/dataframes".format(augumentation.template_path)
#         base_output_dir = loaded_config.count_base_dir.joinpath(
#             augumentation.template_path)
#         iterator = augumentation.make_iterator()
#         # acc_wide = {k: [] for k in range(len(loaded_config.labels))}
#         individual_calculations_dir = base_output_dir.joinpath("distances")
#         train = pd.read_pickle(base_dir + "/0.pickle")
#         train['features'] = train['features'].apply(min_max_scaling)  # data standarization
#         print("Fit metric")
#         # mahalanobis.fit(train)
#         euclidean.fit(train)
#         cosine.fit(train)
#         # k_nearest_neighbours.fit(train)
#         print("Preparations for counting complete")
#         steps = [round(step, 2) for step in iterator]
#         for step in iterator:
#         # process_with_params = partial(perform_step_calculations,
#         #         base_dir=base_dir,
#         #         base_output_dir=base_output_dir, 
#         #         cosine=cosine,
#         #         # mahalanobis=mahalanobis,
#         #         euclidean=euclidean,
#         #         individual_calculations_dir=individual_calculations_dir,
#         #         targets=loaded_config.dataset.targets 
#         #     )
#             perform_step_calculations(step, base_dir, base_output_dir, cosine, euclidean, individual_calculations_dir, loaded_config.dataset.targets)
#         # with Pool(processes=3) as pool:  # Adjust the number of processes as per your machine's capability
#         #     pool.map(process_with_params, steps)
#         #     pool.join()

def perform_step_calculations(
        # distance_method: CosineDistance | MahalanobisDistance | EuclidianDistance,
        step: int, 
        base_dir: str,
        base_output_dir: str, 
        cosine: CosineDistance,
        # mahalanobis: MahalanobisDistance,
        euclidean: EuclidianDistance,
        individual_calculations_dir: pathlib.Path,
        targets: list
):
    step = round(step, 2)
    logging.info("Counting step: {}".format(step))
    file_path = base_dir + "/{}.pickle".format(step)
    if isfile(individual_calculations_dir.joinpath("all-distances-step-{}.pickle".format(step))):
        logging.info("File detected, skipping")
        return
    # output_path_matrix = base_output_dir.joinpath("matrixes/{}".format(step))
    df = pd.read_pickle(file_path)
    df['features'] = df['features'].apply(min_max_scaling)
    # neighbours = k_nearest_neighbours.count_neighbours(df)
    
    # dist_cosine =cosine.count_distance(df)
    dist_euclidean = euclidean.count_distance(df)
    distances = {k: {
        "euclidean": dist_euclidean[k],
        # "cosine":  dist_cosine[k],
        "original_label": df.iloc[k]['original_label'],
        "predicted_label": df.iloc[k]['predicted_label']
    } for k in range(len(targets))
    }
    logging.info("Saving step {}".format(step))
    pd.DataFrame.from_dict(distances, orient="index").to_pickle(
        individual_calculations_dir.joinpath("cosine-euclidean-distances-step-{}.pickle".format(step)))


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
        perform_counting_per_method(conf)
        # count_softmax_values(conf)
        # make_calculations(conf)