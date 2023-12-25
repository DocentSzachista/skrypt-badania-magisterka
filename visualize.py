import seaborn as sn
import numpy as np
from configuration import Config, BASE_PATH, prepare_counted_values_output_dir
from visualization_layer.constants import LABELS_CIFAR_10
import json
import pandas as pd

import pathlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from visualization_layer.calculations import *
import matplotlib.colors as mcolors
from testing_layer.datasets import MixupDataset
from torch.utils.data import DataLoader

from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
from testing_layer.workflows.augumentations import *
import albumentations as A
from testing_layer.custom_transforms import NoiseTransform
from torchvision.utils import save_image


def make_confussion_matrix_plot(cf_matrix: np.ndarray, axs: Axes, plot_title: str, labels: list):

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


def make_step_bar_plot(
        axis: list[Axes],
        fig: Figure, distances: list, save_path: pathlib.Path, title: str, image_class: str, labels: list):
    """
        Zapchaj dziura aby móc łatwiej przemieszczać się w kodzie.
    """
    y_label = "Distance"
    fig.autofmt_xdate(rotation=45)
    axis[0].bar(labels, distances[0], align="center", color=mcolors.TABLEAU_COLORS)
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


def make_visualization_proccess(
        loaded_config: Config, matrix_related=False, individual_images=True, make_average_plots=True):
    # TODO: zoptymalizuj, wyeleminuj powtórzenia, spraw aby ścieżki nie trzeba było co chwila budować
    # matrix_fig, matrix_ax = plt.subplots(1,1, figsize=(10, 10))
    plot_fig, plot_ax = plt.subplots(1, 1, figsize=(12, 12))
    bar_plot_fig, bar_plot_ax = plt.subplots(1, 3, figsize=(12, 12))
    template_title = "Confussion matrix for augumentation {}, augumentation_percentage: {}"
    template_title_total_accuracy = "Total class accuracies for augumentation: {}"
    template_plot_title = "{} distance for image_id: {} of class: {}"

    for augumentation in loaded_config.augumentations:
        base_path_counted = config.count_base_dir.joinpath(BASE_PATH.format(
            loaded_config.model, loaded_config.tag, augumentation.name))

        base_path_visualize = config.visualization_base_dir.joinpath(
            BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name)
        )
        base_path_visualize.mkdir(parents=True, exist_ok=True)
        iterator = augumentation.make_iterator()
        distances = {k: {"mahalanobis": [], "cosine": [], "original_label": [], "predicted_label": []}
                     for k in range(10000)}
        for step in iterator:
            print(f"In step {step}")
            step_percentage = round(100 * step / augumentation.max_size, 2)
            dist_path = base_path_counted.joinpath("distances")
            visul_path = base_path_visualize.joinpath("distances")
            visul_path.mkdir(exist_ok=True)
            step = round(step, 2)
            df = pd.read_pickle(dist_path.joinpath("all-distances-step-{}.pickle".format(round(step, 2))))

            for i in range(len(df.index)):
                if individual_images:
                    image_class = LABELS_CIFAR_10[df.iloc[i]['original_label'][0]]
                    image_class_predicted = LABELS_CIFAR_10[df.iloc[i]['predicted_label'][0]]
                    save_path_bar_plot = visul_path.joinpath(
                        "class-{}-{}/{}".format(df.iloc[i]['original_label'][0], image_class, i))
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
                        ], save_path, super_title, image_class, loaded_config.dataset_labels
                    )
                    distances[i]['mahalanobis'].append(df.iloc[i]['mahalanobis'])
                    distances[i]['cosine'].append(df.iloc[i]['cosine'][0])

        x_axis = np.round(100*iterator / noise_max_range, 2)
        if individual_images:
            for k in distances.keys():
                np_array = np.asarray(distances[k]['mahalanobis'])
                # print(np_array.shape)
                image_class_id = df.iloc[k]['original_label'][0]
                image_class = LABELS_CIFAR_10[image_class_id]
                # for i in range(np_array.shape[1]):
                plot_ax.plot(x_axis, np_array[:, 0], label=LABELS_CIFAR_10.values())
                plot_ax.set_title(template_plot_title.format("Mahalanobis", k, LABELS_CIFAR_10[image_class_id]),)
                plot_ax.set_xlabel("percentage image augumentation")
                plot_ax.set_ylabel("Distance")
                plot_ax.grid()
                plot_ax.legend()
                save_path = base_path_visualize.joinpath(
                    "distances/class-{}-{}/{}".format(image_class_id, image_class, k))
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
                # 336
                # step_path = base_path_visualize.joinpath(step)
                # step_path.mkdir(parents=False, exist_ok=True)

                # np.load(path_to_matrix)


def generate_step_images(conf: Config):
    transform = A.Compose([
        ToTensorV2()
    ])

    # cifar = CIFAR10("./datasets", train=False, download=True,
    # transform=lambda x: transform(image=np.array(x))["image"].float()/255.0)
    # cat_class_indices = [idx for idx, label in enumerate(cifar.targets) if label == 3]
    path = pathlib.Path("visualizations")
    for augumentation in conf.augumentations:
        save_path = pathlib.Path(
            "./visualizations/{}".format(BASE_PATH.format(conf.model, conf.tag, augumentation.name)))
        save_path = save_path.joinpath("images")
        save_path.mkdir(exist_ok=True)
        print("current augumentation {}".format(augumentation.name))
        iterator = augumentation.make_iterator()
        for step in iterator:
            step_percentage = round(100 * step / augumentation.max_size, 2)
            print("current step {}".format(step))
            if isinstance(augumentation, MixupAugumentation):
                indices = [idx for idx, label
                           in enumerate(conf.dataset.targets)
                           if label == augumentation.class_
                           ]
                dataset_step = MixupDataset(conf.dataset, indices, step,
                                            should_save_processing=conf.save_preprocessing, path=save_path)
                dataloader = DataLoader(dataset_step, batch_size=50, shuffle=False, drop_last=False)
            elif isinstance(augumentation, NoiseAugumentation):
                transforms = A.Compose([
                    NoiseTransform(
                        number_of_pixels=step, shuffled_indexes=augumentation.shuffled_indexes,
                        mask=augumentation.mask, image_len=conf.image_dim[1]),
                    ToTensorV2()])
                conf.dataset.transform = lambda x: transforms(image=np.array(x))["image"].float()/255.0
                dataloader = DataLoader(conf.dataset, batch_size=1, shuffle=False, drop_last=False)
            for id, (inputs, targets) in enumerate(dataloader):
                if augumentation.name == "noise":
                    singular = save_path.joinpath(f"{targets[0]}/{id}")
                    singular.mkdir(parents=True, exist_ok=True)
                    save_image(inputs,  singular.joinpath(f"noised_sample-{id}-step-{step_percentage}.jpg"))


def make_matrix_plots(loaded_config: Config, labels: list):
    """Tworzy wykresy macierzy pomyłek oraz wykres zmianny accuracy dla danych klas"""
    template_title = "Confussion matrix for augumentation {}, augumentation_percentage: {}"
    for augumentation in loaded_config.augumentations:
        base_path_counted = config.count_base_dir.joinpath(augumentation.template_path)
        base_path_visualize = config.visualization_base_dir.joinpath(
            augumentation.template_path
        )

        base_path_visualize.mkdir(parents=True, exist_ok=True)
        iterator = augumentation.make_iterator()
        storage = []
        x_axis = np.round(100*iterator / augumentation.max_size, 2)
        for step in iterator:
            print(f"In step {step}")
            step_percentage = round(100 * step / augumentation.max_size, 2)
            matrix_fig, matrix_ax = plt.subplots(1, 1, figsize=(10, 10))
            path_to_matrix = base_path_counted.joinpath("matrixes/{}.npy".format(round(step, 2)))
            cf_matrix = np.load(path_to_matrix)
            make_confussion_matrix_plot(
                cf_matrix,
                matrix_ax,
                template_title.format(augumentation.name, step_percentage),
                labels=loaded_config.dataset_labels
            )
            matrix_fig.savefig(base_path_visualize.joinpath("confussion_matrix-step{}.png".format(step)))

            arr = np.empty(shape=(10))
            for i in range(cf_matrix.shape[0]):
                arr[i] = cf_matrix[i, i] / np.sum(cf_matrix[i, :])
            storage.append(arr)
        storage = np.asarray(storage)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle("Change of accuracies for augumentation: {}".format(augumentation.name))
        ax.set_xlabel("Percentage of noise")
        ax.set_ylabel("Accuracy")
        for index in range(len(labels)):
            ax.plot(x_axis, storage[:, index],  label=labels[index])
        ax.legend()
        ax.grid()
        fig.savefig(base_path_visualize.joinpath("accuracies_plot.png"))


def make_average_plots(loaded_config: Config, labels: list):
    """Tworzy wykresy średnich odległości i liczby sąsiadów w zależności od klasy"""
    def prepare_axis(ax: plt.Axes, x_axis: list, y_axis: np.ndarray, title: str, x_label: str, y_label: str):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        return ax.plot(
            x_axis, y_axis
        )

    x_label = "percentage image augumentation"
    dist_label = "Distance"
    neigh_label = "number of neightbours"

    for augumentation in loaded_config.augumentations:
        base_path_counted = config.count_base_dir.joinpath(BASE_PATH.format(
            loaded_config.model, loaded_config.tag, augumentation.name))

        base_path_visualize = config.visualization_base_dir.joinpath(
            BASE_PATH.format(loaded_config.model, loaded_config.tag, augumentation.name)
        )

        base_path_visualize.mkdir(parents=True, exist_ok=True)
        iterator = augumentation.make_iterator()
        mean_distances = {k: {"mahalanobis": [], "cosine": [], "euclidean": [], "neighbours": []} for k in range(10)}
        x_axis = np.round(100*iterator / augumentation.max_size, 2)

        dist_path = base_path_counted.joinpath("distances")
        distance_visul_path = base_path_visualize.joinpath("distances")
        distance_visul_path.mkdir(exist_ok=True)
        neightbours_visul_path = base_path_visualize.joinpath("neightbours")
        neightbours_visul_path.mkdir(exist_ok=True)

        # Część gdzie podliczane są średnie odległości wobec klasy obrazków oraz średnia liczba
        # sąsiadów dla danego stopnia zaszumienia
        for step in iterator:
            print(f"In step {step}")
            step = round(step, 2)
            df = pd.read_pickle(dist_path.joinpath("all-distances-step-{}.pickle".format(step)))
            by_state = df.groupby("original_label")

            for state, frame in by_state:
                mean_distances[state]['mahalanobis'].append(
                    frame.mahalanobis.mean()
                )
                mean_distances[state]['euclidean'].append(
                    frame.euclidean.mean()
                )
                mean_distances[state]['cosine'].append(
                    frame.cosine.mean()
                )
                mean_distances[state]['neighbours'].append(
                    frame.neighbours.apply(pd.Series).mean().values
                )
        # część gdzie podliczane są tworzone wykresy
        for class_, values in mean_distances.items():
            plot_fig, plot_ax = plt.subplots(1, 3, figsize=(15, 10))
            plot_fig.suptitle("Average distances for images labeled: {} from others by using {} augumentation".format(
                labels[class_], augumentation.name))

            conv_mahal = np.vstack(values['mahalanobis'])
            conv_cosine = np.vstack(values['cosine'])
            conv_euclid = np.vstack(values['euclidean'])
            conv_neight = np.vstack(values['neighbours'])
            l1 = prepare_axis(plot_ax[0], x_axis, conv_mahal, "mahalanobis",
                              x_label, dist_label,)
            prepare_axis(plot_ax[1], x_axis, conv_cosine, "cosine", x_label, dist_label,)
            prepare_axis(plot_ax[2], x_axis, conv_euclid, "euclidean", x_label, dist_label,)
            plot_fig.legend(
                tuple(l1), tuple(labels),  loc='outside right upper'
            )
            plot_fig.savefig(distance_visul_path.joinpath("mean-distances-{}.png".format(labels[class_])))

            plot_fig, plot_ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_fig.suptitle(
                "Average number of neighbours for images labeled: {} per percentage\n {} augumentation".format(
                    labels[class_],
                    augumentation.name))
            line = prepare_axis(
                plot_ax, x_axis, conv_neight, "", x_label, neigh_label
            )
            plot_fig.legend(
                tuple(line), tuple(labels),  loc='outside right upper'
            )
            plot_fig.savefig(neightbours_visul_path.joinpath("mean-neighbours-{}.png".format(labels[class_])))


if __name__ == "__main__":
    with open("./config.json", "r") as file:
        obj = json.load(file)
    config = Config(obj)
    prepare_counted_values_output_dir(config)
    sn.set_theme()
    # make_average_plots(config, config.dataset_labels)
    # make_matrix_plots(config, config.dataset_labels)
    generate_step_images(config)
