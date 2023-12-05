from configuration import Config, BASE_PATH, prepare_counted_values_output_dir 
from visualization_layer.constants import LABELS_CIFAR_10
import json 
import pandas as pd
from visualization_layer.plots import confussion
import pathlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from visualization_layer.calculations import *
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from testing_layer.datasets import MixupDataset
from torch.utils.data import DataLoader

from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
from testing_layer.workflows.augumentations import *
import albumentations as A
from testing_layer.custom_transforms import NoiseTransform
from torchvision.utils import save_image



def make_calculations(loaded_config: Config):
    """ Przeprowadź obliczenia odległości, i innych statystyk na podstawie konfiguracji. """
    train = pd.read_pickle("./test_cifar10.pickle")
    cosine =  CosineDistance()
    mahalanobis = MahalanobisDistance()
    euclidean = EuclidianDistance()
    train['features'] = train['features'].apply(lambda x: (x - np.mean(x) ) /np.std(x))
    mahalanobis.fit(train)
    
    for augumentation in loaded_config.augumentations:
        base_dir = "{}/dataframes".format(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name))
        base_output_dir =  pathlib.Path("./counted_outputs/{}".format(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name)))
        iterator = augumentation.make_iterator()
        acc_wide = { k: [] for k in LABELS_CIFAR_10.keys()}
        print("Preparations for counting complete")
        for step in iterator:
            step = round(step, 2)
            distances = {k: {"mahalanobis": [], "cosine": [], "euclidean": [], "original_label": [], "predicted_label": []} for k in range(10000)}
            print("Counting step: {}".format(step))
            file_path = base_dir + "/{}.pickle".format(step)
            output_path_matrix = base_output_dir.joinpath("matrixes/{}".format(step))
            df = pd.read_pickle(file_path)
            df['features'] = df['features'].apply(lambda x: (x - np.mean(x) ) /np.std(x))
            y_pred = df['predicted_label'].values
            y_true = df['original_label'].values
            confussion.make_noise_wide_statistic(y_true, y_pred, list(LABELS_CIFAR_10.keys()), acc_wide, filename=output_path_matrix)
            dist_mahalanobis = mahalanobis.count_distance(df).T
            for i in range(len(train.index)):
                dist_cosine = cosine.count_distance(train.iloc[i]['features'], df.iloc[i]['features'])
                euc = euclidean.count_distance(train.iloc[i]['features'], df.iloc[i]['features'])
                distances[i]['cosine'].append(dist_cosine)
                distances[i]['mahalanobis'].append(list(dist_mahalanobis[i]))
                distances[i]['euclidean'].append(euc)
                distances[i]['original_label'].append(df.iloc[i]['original_label'])
                distances[i]['predicted_label'].append(df.iloc[i]['predicted_label'])
            individual_calculations_dir = base_output_dir.joinpath("distances")            
            pd.DataFrame.from_dict(distances, orient="index").to_pickle(individual_calculations_dir.joinpath("all-distances-step-{}.pickle".format(step)))
        # with open(individual_calculations_dir.joinpath("noise_steps.json"), "w+") as file: 
        #     json.dump(list(iterator), file)
        with open(base_output_dir.joinpath("classes-accuracy.json"), "w+") as file: 
            json.dump(acc_wide, file)
            


def make_line_plot(source_file_path: str, iterator: list, axs: Axes, plot_title: str):
    with open(source_file_path, "r+") as file: 
        data = json.load(file)
    for key, values in data.items():
        axs.plot(iterator, values,  label=LABELS_CIFAR_10[int(key)])
    axs.set_xlabel("Stopień zaszumienia")
    axs.set_ylabel("Accuracy")
    axs.set_title("Rodzaj ingerencji w obraz {}".format(plot_title))
    axs.legend()
    axs.grid()
        

import numpy as np 
import seaborn as sn
def make_confussion_matrix_plot(source_file_path: str, axs: Axes, plot_title: str, labels: list):
    
    cf_matrix = np.load(source_file_path)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in labels],
        columns=[i for i in labels]
    )    
    axs.set_title(plot_title)
    axs.set_xlabel("predicted")
    axs.set_ylabel("Actual")
    sn.heatmap(
        df_cm, annot=True, fmt="g", ax=axs
    )

def debug_func(iterator: list, labels: list, title: str, read_path: pathlib.Path, save_path: pathlib.Path, max_range: int):
    storage = []
    for step in iterator: 
        cf_matrix = np.load(read_path.joinpath("matrixes/{}.npy".format(round(step, 2))))
        arr = np.empty(shape=(10))
        for i in range(cf_matrix.shape[0]):
            arr[i] = cf_matrix[i, i] / np.sum(cf_matrix[i, :])
        storage.append(arr)
    storage = np.asarray(storage)
    print(storage)
    print(storage.shape)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle(title)
    ax.set_xlabel("Percentage of noise")
    ax.set_ylabel("Accuracy")
    for index in range(len(labels)):
        ax.plot(np.round(100*iterator / max_range, 2), storage[:, index],  label=labels[index])
    ax.legend()
    ax.grid()
    fig.savefig(save_path)

def make_statistics_per_image(augumentation, df: pd.DataFrame, iterator: list, base_path: pathlib.Path):
    """Stwórz statystyki co pojedynczy obrazek."""
    x_label = "Noise percentage"
    y_label = "Distance"
    aug_string = augumentation.name
    for i, row in df.iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(10,10))
        fig.suptitle("Distance changes for image id: {} class of {}\nfrom{}".format(i, LABELS_CIFAR_10[df.iloc[i]['original_label']]))
        axes[0].set_title("Mahalanobis distance change")
        axes[1].set_title("Cosine distance change")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel(y_label)
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel(y_label)
        converted_mahalanobis = np.asarray(row['mahalanobis'])
        for sub_array, key in zip(converted_mahalanobis.T, list(LABELS_CIFAR_10.keys())):
            axes[0].plot( round(100 * (iterator / 1024), 2), sub_array, label=LABELS_CIFAR_10[key])
        axes[1].plot(row['iterator'], row['cosine'], label="distance from {}".format(aug_string))
        axes[0].legend()
        axes[1].legend()
        
        fig.savefig(base_path.joinpath("id_{}_label_{}.png".format(i, LABELS_CIFAR_10[df.iloc[i].original_label])))


def make_step_bar_plot(axis : list[Axes], fig: Figure, distances: list, save_path: pathlib.Path, title: str, image_class: str ):
    """
        Zapchaj dziura aby móc łatwiej przemieszczać się w kodzie. 
    """
    y_label = "Distance"
    fig.autofmt_xdate(rotation=45)
    axis[0].bar(list(LABELS_CIFAR_10.values()), distances[0], align="center", color=mcolors.TABLEAU_COLORS)
    axis[0].set_ylabel(y_label)
    axis[0].set_title("Mahalanobis distance")
    axis[1].set_ylim(bottom=0, top=1)
    axis[1].set_ylabel(y_label)
    axis[1].bar(image_class, distances[1], align="center")
    axis[1].set_title("Cosine distance")

    fig.suptitle(title)
    axis[2].bar(image_class, distances[2], align="center")                
    axis[2].set_title("Euclidean distance")
    axis[2].set_ylabel(y_label)
        
    fig.savefig(save_path)
    axis[0].cla()
    axis[1].cla()
    axis[2].cla()




def make_visualization_proccess(loaded_config: Config, matrix_related=False, individual_images=True):
    # TODO: zoptymalizuj, wyeleminuj powtórzenia, spraw aby ścieżki nie trzeba było co chwila budować
    # matrix_fig, matrix_ax = plt.subplots(1,1, figsize=(10, 10))
    plot_fig, plot_ax = plt.subplots(1,1, figsize=(12, 12))
    bar_plot_fig, bar_plot_ax = plt.subplots(1, 3, figsize=(12,12))
    template_title = "Confussion matrix for augumentation {}, augumentation_percentage: {}"
    template_title_total_accuracy = "Total class accuracies for augumentation: {}"
    template_plot_title = "{} distance for image_id: {} of class: {}"    

    for augumentation in loaded_config.augumentations:
        base_path_counted =  pathlib.Path("./counted_outputs/{}".format(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name)))
        base_path_visualize = pathlib.Path("./visualizations/{}".format(BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name)))
        base_path_visualize.mkdir(parents=True, exist_ok=True)
        iterator = augumentation.make_iterator()
        # if matrix_related:
        #     make_line_plot(base_path_counted.joinpath("classes-accuracy.json"), iterator, plot_ax, template_title_total_accuracy.format(augumentation.name))
        #     plot_fig.savefig(base_path_visualize.joinpath("classes-accuracy.png"))
        #     plot_ax.cla()
        # plot_fig.clear()
        noise_max_range = 1024 if augumentation.name == "noise" else 100
        # debug_func(iterator, list(LABELS_CIFAR_10.values()), template_title_total_accuracy.format(augumentation.name),
        #             base_path_counted, base_path_visualize.joinpath("accuracies_plot.png"), noise_max_range )
        distances = {k: {"mahalanobis": [], "cosine": [], "original_label": [], "predicted_label": []} for k in range(10000)}
       
        for step in iterator:
            matrix_fig, matrix_ax = plt.subplots(1,1, figsize=(10, 10))
            path_to_matrix = base_path_counted.joinpath("matrixes/{}.npy".format(round(step, 2)))
            noise_max_range = 1024 if augumentation.name == "noise" else 100
            step_percentage = round(100 * step / noise_max_range, 2  )
            if matrix_related:
                make_confussion_matrix_plot(
                    path_to_matrix, matrix_ax, 
                    template_title.format(augumentation.name, step_percentage), 
                    labels=list(LABELS_CIFAR_10.values())
                )
                matrix_fig.savefig(base_path_visualize.joinpath("confussion_matrix-step{}.png".format(step)))
                # matrix_ax.cla()
            
            dist_path = base_path_counted.joinpath("distances")
            visul_path = base_path_visualize.joinpath("distances")
            visul_path.mkdir(exist_ok=True)
            step = round(step, 2)
            df = pd.read_pickle(dist_path.joinpath("all-distances-step-{}.pickle".format(round(step, 2))))
            print(f"In step {step}")
            for i in range(len(df.index)):
                if individual_images:
                    image_class = LABELS_CIFAR_10[df.iloc[i]['original_label'][0]]
                    image_class_predicted = LABELS_CIFAR_10[df.iloc[i]['predicted_label'][0]]
                    save_path_bar_plot = visul_path.joinpath("class-{}-{}/{}".format(df.iloc[i]['original_label'][0], image_class, i ))
                    save_path_bar_plot.mkdir(parents=True, exist_ok=True)
                    # save_path_bar_plot.joinpath(f"{i}").mkdir(exist_ok=True)
                    save_path = save_path_bar_plot.joinpath("barplot-percentage-{}.png".format(step))
                    super_title = "Image id: {} class origin: {} class predicted: {} \n augumentation %: {}%".format(
                        i, image_class, image_class_predicted, step_percentage)
                    make_step_bar_plot(
                        bar_plot_ax, bar_plot_fig, [
                            df.iloc[i]['mahalanobis'][0],
                            df.iloc[i]['cosine'][0],
                            df.iloc[i]['euclidean'][0]  
                        ], save_path, super_title, image_class
                    )
                    distances[i]['mahalanobis'].append(df.iloc[i]['mahalanobis'])
                    distances[i]['cosine'].append(df.iloc[i]['cosine'][0])
                # distances[i]['euclidean'].extend(df.iloc[i]['euclidean'])
            # distances[i]['original_label'].append(df.iloc[i]['original_label'][0])
            # distances[i]['predicted_label'].append(df.iloc[i]['predicted_label'][0])
        x_axis = np.round(100*iterator / noise_max_range, 2)      
        if individual_images:  
            for k in distances.keys():
                np_array = np.asarray(distances[k]['mahalanobis']) 
                # print(np_array.shape)
                image_class_id = df.iloc[k]['original_label'][0]
                image_class = LABELS_CIFAR_10[image_class_id]
                # for i in range(np_array.shape[1]):
                plot_ax.plot( x_axis, np_array[:, 0], label=LABELS_CIFAR_10.values())
                plot_ax.set_title(template_plot_title.format("Mahalanobis", k, LABELS_CIFAR_10[image_class_id]),)
                plot_ax.set_xlabel("percentage image augumentation")
                plot_ax.set_ylabel("Distance")
                plot_ax.grid()
                plot_ax.legend()
                save_path = base_path_visualize.joinpath("distances/class-{}-{}/{}".format(image_class_id, image_class, k))            
                save_path.mkdir(parents=True, exist_ok=True)
                plot_fig.savefig(save_path.joinpath("mahalanobis-dist.png"))
                # break
                plot_ax.cla()
                # Cosinusowa 
                plot_ax.plot(x_axis, distances[k]['cosine'], label=image_class)
                plot_ax.set_title(template_plot_title.format("Cosine", k, image_class),)
                plot_ax.set_xlabel(f"percentage image augumentation of type: {augumentation.name} ")
                plot_ax.set_ylabel("Distance")
                plot_ax.grid()
                plot_fig.savefig(save_path.joinpath("cosine-dist.png"))
                plot_ax.cla()
                # euklidesowa
                #336
                # step_path = base_path_visualize.joinpath(step)
                # step_path.mkdir(parents=False, exist_ok=True)
                
                # np.load(path_to_matrix)
def make_overall_dist_plot_per_image(ax : Axes, fig: Figure, title: str, x_axis: list, y_axis: np.ndarray, save_path: str ,  label: str | list):
    
    ax.plot( x_axis, y_axis, label=LABELS_CIFAR_10.values())
    # template_plot_title.format("Mahalanobis", k, LABELS_CIFAR_10[image_class_id])
    ax.set_title(title)
    ax.set_xlabel("percentage image augumentation")
    ax.set_ylabel("Distance")
    ax.grid()
    ax.legend()
    # save_path = base_path_visualize.joinpath("distances/class-{}-{}/{}".format(image_class_id, image_class, k))            
    # save_path.mkdir(parents=True, exist_ok=True)
    # save_path.joinpath("mahalanobis-dist.png")
    # plot_fig.savefig(save_path)
    # # break
    # plot_ax.cla()


def generate_step_images(conf: Config):
    transform = A.Compose([
            ToTensorV2()
        ])
    
    cifar = CIFAR10("./datasets", train=False,  transform=lambda x: transform(image=np.array(x))["image"].float()/255.0 )
    cat_class_indices = [idx for idx, label in enumerate(cifar.targets) if label == 3]
    path = pathlib.Path("visualizations")  
    for augumentation in conf.augumentations: 
        save_path = pathlib.Path("./visualizations/{}".format(BASE_PATH.format(conf.model, conf.tag, augumentation.name)))
        save_path = save_path.joinpath("images")
        save_path.mkdir(exist_ok=True)
        formatted_path = BASE_PATH.format(conf.model, conf.tag, augumentation.name)
        print("current augumentation {}".format(augumentation.name))
        iterator = augumentation.make_iterator()
        noise_max_range = 1024 if augumentation.name == "noise" else 100
        for step in iterator: 
            step_percentage = round(100 * step / noise_max_range, 2  )
            print("current step {}".format(step))            
            if isinstance(augumentation, MixupAugumentation): 
                dataset_step = MixupDataset(cifar, cat_class_indices, step, should_save_processing=conf.save_preprocessing, path=save_path )
                dataloader =  DataLoader(dataset_step, batch_size=50, shuffle=False, drop_last=False)
            elif isinstance(augumentation, NoiseAugumentation):
                transforms = A.Compose([
                    NoiseTransform(number_of_pixels=step, shuffled_indexes=augumentation.shuffled_indexes, mask=augumentation.mask),
                    ToTensorV2()
                ])
                cifar.transform = lambda x : transforms(image=np.array(x))["image"].float()/255.0 
                dataloader = DataLoader(cifar, batch_size=1, shuffle=False, drop_last=False)
            for id, (inputs, targets) in enumerate(dataloader):
                if augumentation.name == "noise":
                    singular = save_path.joinpath(f"{targets[0]}/{id}")
                    singular.mkdir(parents=True, exist_ok=True)
                    save_image(inputs,  singular.joinpath(f"noised_sample-{id}-step-{step_percentage}.jpg") )
                    # print("here")
    
    



if __name__ == "__main__": 
    with open("./config.json", "r") as file:
        obj = json.load(file)
    config = Config(obj)
    prepare_counted_values_output_dir(config)
    # make_calculations(config)
    make_visualization_proccess(config, individual_images=False, matrix_related=True)
    # generate_step_images(config)
    # iterator = config.augumentations[0].make_iterator()
    # percentage = [round(100 * step / 1024, 3) for step in iterator]
    # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # make_line_plot("./counted_outputs/resnet-red/noise/classes-accuracy.json", percentage, axs, config.augumentations[0].name )
    








    
