from configuration import Config, BASE_PATH, prepare_counted_values_output_dir 
from visualization_layer.constants import LABELS_CIFAR_10
import json 
import pandas as pd
from visualization_layer.plots import confussion
import pathlib
from visualization_layer.calculations import *
from testing_layer.workflows.augumentations import *

def handle_mixup(augumentation: MixupAugumentation,  loaded_config: Config, k_nearest_neighbours: NearestNeightboursCount, 
                 cosine: CosineDistance, mahalanobis: MahalanobisDistance, euclidean: EuclidianDistance): 
    print("perform calculations for {}".format(augumentation.name))
    for chosen_class in augumentation.classes:
        base_dir = "{}/dataframes/mixup_to_{}".format(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name), chosen_class)
        base_output_dir = loaded_config.count_base_dir.joinpath(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name))
        base_output_dir = base_output_dir.joinpath("mixup_to_{}".format(chosen_class))
        base_output_dir.mkdir(exist_ok=True)
        iterator = augumentation.make_iterator()
        acc_wide = { k: [] for k in LABELS_CIFAR_10.keys()}
        print("Preparations for counting complete")
        for step in iterator:
            step = round(step, 2)
            print("Counting step: {}".format(step))
            file_path = base_dir + "/{}.pickle".format(step)
            output_path_matrix = base_output_dir.joinpath("matrixes/{}".format(step))
            df = pd.read_pickle(file_path)
            df['features'] = df['features'].apply(lambda x: (x - np.mean(x) ) /np.std(x))
            y_pred = df['predicted_label'].values
            y_true = df['original_label'].values
            confussion.make_noise_wide_statistic(y_true, y_pred, list(LABELS_CIFAR_10.keys()), acc_wide, filename=output_path_matrix)
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
                } for k in range(10000)
            }
            individual_calculations_dir = base_output_dir.joinpath("distances")            
            pd.DataFrame.from_dict(distances, orient="index").to_pickle(individual_calculations_dir.joinpath("all-distances-step-{}.pickle".format(step)))
        # with open(base_output_dir.joinpath("classes-accuracy.json"), "w+") as file: 
        #     json.dump(acc_wide, file)

def handle_noise(augumentation: NoiseAugumentation,  loaded_config: Config, k_nearest_neighbours: NearestNeightboursCount, 
                 cosine: CosineDistance, mahalanobis: MahalanobisDistance, euclidean: EuclidianDistance):
    print("perform calculations for {}".format(augumentation.name))
    base_dir = "{}/dataframes".format(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name))
    base_output_dir = loaded_config.count_base_dir.joinpath(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name))
    iterator = augumentation.make_iterator()
    acc_wide = { k: [] for k in LABELS_CIFAR_10.keys()}
    print("Preparations for counting complete")
    for step in iterator:
        step = round(step, 2)
        print("Counting step: {}".format(step))
        file_path = base_dir + "/{}.pickle".format(step)
        output_path_matrix = base_output_dir.joinpath("matrixes/{}".format(step))
        df = pd.read_pickle(file_path)
        df['features'] = df['features'].apply(lambda x: (x - np.mean(x) ) /np.std(x))
        y_pred = df['predicted_label'].values
        y_true = df['original_label'].values
        confussion.make_noise_wide_statistic(y_true, y_pred, list(LABELS_CIFAR_10.keys()), acc_wide, filename=output_path_matrix)
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
            } for k in range(10000)
        }
        individual_calculations_dir = base_output_dir.joinpath("distances")            
        pd.DataFrame.from_dict(distances, orient="index").to_pickle(individual_calculations_dir.joinpath("all-distances-step-{}.pickle".format(step)))
    with open(base_output_dir.joinpath("classes-accuracy.json"), "w+") as file: 
        json.dump(acc_wide, file)
            



def make_calculations(loaded_config: Config):
    """ Przeprowadź obliczenia odległości, i innych statystyk na podstawie konfiguracji. """
    train = pd.read_pickle("./cifar_10.pickle")
    cosine =  CosineDistance()
    mahalanobis = MahalanobisDistance()
    euclidean = EuclidianDistance()
    k_nearest_neighbours = NearestNeightboursCount(100)
    
    train['features'] = train['features'].apply(lambda x: (x - np.mean(x) ) /np.std(x)) # data standarization
    # prepare train features for distance counting 
    mahalanobis.fit(train)
    euclidean.fit(train)
    cosine.fit(train)
    k_nearest_neighbours.fit(train)
    for augumentation in loaded_config.augumentations:
        if isinstance(augumentation, NoiseAugumentation):
            handle_noise(augumentation, loaded_config, k_nearest_neighbours, cosine, mahalanobis, euclidean)
        elif isinstance(augumentation, MixupAugumentation):
            handle_mixup(augumentation, loaded_config, k_nearest_neighbours, cosine, mahalanobis, euclidean)    
            
if __name__ == "__main__":
    with open("./config.json", "r") as file:
        obj = json.load(file)
    conf = Config(obj)
    prepare_counted_values_output_dir(conf)
    make_calculations(conf)