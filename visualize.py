import seaborn as sn
import numpy as np
from testing_layer.configuration import Config, BASE_PATH, prepare_counted_values_output_dir, prepare_visualization_output_dir
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
from testing_layer.workflows.enums import SupportedModels
from testing_layer.datasets import MixupDataset, ImageNetKaggle




X_PERCENT_LABEL = "nałożona modyfikacja (%)"
Y_SOFTMAX_LABEL = "klasy"
SOFMTAX_TITLE = "Wykres zmiany SOFTMAX dla: {}"

X_LABEL_CONFUSSION = "Actual"
Y_LABEL_CONFUSSION = "Predicted"

Y_DIST_LABEL = "Distance"
NEIGH_LABEL = "number of neightbours"





def make_confussion_matrix_plot(cf_matrix: np.ndarray, axs: Axes, plot_title: str, labels: list):

    df_cm = pd.DataFrame(
        cf_matrix, #/ np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in labels],
        columns=[i for i in labels]
    )
    axs.set_title(plot_title)
    axs.set_xlabel(X_LABEL_CONFUSSION)
    axs.set_ylabel(Y_LABEL_CONFUSSION)
    sn.heatmap(
        df_cm, annot=True, fmt="g", ax=axs
    )


def prepare_axis(ax: plt.Axes, x_axis: list, y_axis: np.ndarray,
                     title: str, x_label: str, y_label: str):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid()
        return ax.plot(
            x_axis, y_axis,
        )





def make_step_bar_plot(
        axis: list[Axes],
        fig: Figure, distances: list, save_path: pathlib.Path, title: str, image_class: str, labels: list):
    """
        Zapchaj dziura aby móc łatwiej przemieszczać się w kodzie.
    """
    y_label = "Distance"
    fig.autofmt_xdate(rotation=45)
    axis[0].bar(labels, distances[0], align="center", color=mcolors.TABLEAU_COLORS,)
    axis[0].set_ylabel(y_label)
    axis[0].set_title("Mahalanobis distance")
    axis[1].set_ylim(bottom=0, top=1)
    axis[1].set_ylabel(y_label)
    axis[1].bar(labels, distances[1], align="center", color=mcolors.TABLEAU_COLORS)
    axis[1].set_title("Cosine distance")

    fig.suptitle(title)
    axis[2].bar(labels, distances[2], align="center", color=mcolors.TABLEAU_COLORS)
    axis[2].set_title("Euclidean distance")
    axis[2].set_ylabel(y_label)

    fig.savefig(save_path)
    axis[0].cla()
    axis[1].cla()
    axis[2].cla()


def make_individual_stats(
        loaded_config: Config, individual_images=True, ):
    # TODO: zoptymalizuj, wyeleminuj powtórzenia, spraw aby ścieżki nie trzeba było co chwila budować
    # matrix_fig, matrix_ax = plt.subplots(1,1, figsize=(10, 10))
    plot_fig, plot_ax = plt.subplots(1, 1, figsize=(12, 12))
    bar_plot_fig, bar_plot_ax = plt.subplots(1, 3, figsize=(12, 12))
    template_plot_title = "{} distance for image_id: {} of class: {}"

    for augumentation in loaded_config.augumentations:
        base_path_counted = config.count_base_dir.joinpath(augumentation.template_path)

        base_path_visualize = config.visualization_base_dir.joinpath(
           augumentation.template_path
        )
        base_path_visualize.mkdir(parents=True, exist_ok=True)
        iterator = augumentation.make_iterator()
        distances = {k: {"mahalanobis": [], "cosine": [], "original_label": [], "euclidean":[], "predicted_label": []}
                     for k in range(len(loaded_config.dataset))}
        for step in iterator:
            print(f"In step {step}")
            step_percentage = round(100 * step / augumentation.max_size, 2)
            dist_path = base_path_counted.joinpath("distances")
            visul_path = base_path_visualize.joinpath("distances")
            visul_path.mkdir(exist_ok=True)
            step = round(step, 2)
            df = pd.read_pickle(dist_path.joinpath("all-distances-step-{}.pickle".format(round(step, 2))))

            for i in range(len(df.index)):
                image_class = loaded_config.labels[df.iloc[i]['original_label']]
                image_class_predicted = loaded_config.labels[df.iloc[i]['predicted_label']]
                save_path_bar_plot = visul_path.joinpath(
                    "class-{}-{}/{}".format(df.iloc[i]['original_label'], image_class, i))
                save_path_bar_plot.mkdir(parents=True, exist_ok=True)
                # save_path_bar_plot.joinpath(f"{i}").mkdir(exist_ok=True)
                save_path = save_path_bar_plot.joinpath("barplot-percentage-{}.png".format(step))
                super_title = "Image id: {} class origin: {} class predicted: {} \n augumentation %: {}%".format(
                    i, image_class, image_class_predicted, step_percentage)
                make_step_bar_plot(
                    bar_plot_ax, bar_plot_fig, [
                        df.iloc[i]['mahalanobis'],
                        df.iloc[i]['cosine'],
                        df.iloc[i]['euclidean']
                    ], save_path, super_title, image_class, loaded_config.labels
                )
                distances[i]['mahalanobis'].append(df.iloc[i]['mahalanobis'])
                distances[i]['cosine'].append(df.iloc[i]['cosine'])
                distances[i]['euclidean'].append(df.iloc[i]['euclidean'])

        x_axis = np.round(100*iterator / augumentation.max_size, 2)
        if individual_images:
            for k in distances.keys():
                np_array = np.asarray(distances[k]['mahalanobis'])
                # print(np_array.shape)
                image_class_id = df.iloc[k]['original_label'][0]
                image_class = loaded_config.labels[image_class_id]
                # for i in range(np_array.shape[1]):
                plot_ax.plot(x_axis, np_array[:, 0], label=loaded_config.labels)
                plot_ax.set_title(template_plot_title.format("Mahalanobis", k, loaded_config.labels[image_class_id]),)
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
                plot_ax.plot(x_axis, distances[k]['euclidean'], label=image_class)
                plot_ax.set_title(template_plot_title.format("Euclidean", k, image_class),)
                plot_ax.set_xlabel(f"percentage image augumentation of type: {augumentation.name} ")
                plot_ax.set_ylabel("Distance")
                plot_ax.grid()
                plot_fig.savefig(save_path.joinpath("euclidean-dist.png"))
                plot_ax.cla()


def generate_step_images(conf: Config):
    transform = A.Compose([
        ToTensorV2()
    ])

    path = pathlib.Path("visualizations")
    for augumentation in conf.augumentations:
        save_path = pathlib.Path(
            "./visualizations/{}".format(augumentation.template_path))
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
    thresholds = {}
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
            step = round(step, 2)
            print(f"In step {step}")
            step_percentage = round(100 * step / augumentation.max_size, 2)
            matrix_fig, matrix_ax = plt.subplots(1, 1, figsize=(10, 10))
            path_to_matrix = base_path_counted.joinpath("matrixes/{}.npy".format(round(step, 2)))
            cf_matrix = np.load(path_to_matrix)

            make_confussion_matrix_plot(
                cf_matrix,
                matrix_ax,
                template_title.format(augumentation.name, step_percentage),
                labels=labels
            )
            matrix_fig.savefig(base_path_visualize.joinpath("confussion_matrix-step{}.png".format(step)))

            arr = np.empty(shape=(len(loaded_config.labels)))
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

        thresholds[augumentation.template_path] = np.argmax(storage < 0.5, axis=0)

    return thresholds



def make_average_softmax_plot(config: Config):
    """ Sporządź średni wykres funkcji softmax dla logitsów.

    """
    figure, axis = plt.subplots(1, 1, figsize=(20, 20))

    for augumentation in config.augumentations:
        base_path_counted = config.count_base_dir.joinpath(augumentation.template_path)
        iterator = augumentation.make_iterator()
        base_path_visualize = config.visualization_base_dir.joinpath(
           augumentation.template_path
        ).joinpath("softmaxes")

        base_path_visualize.mkdir(parents=True, exist_ok=True)
        x_axis = np.round(100*iterator / augumentation.max_size, 2)
        softmax_means = {k: [] for k in range(len(config.labels))}
        softmax_all = {k: [] for k in range(len(config.labels))} 
        dist_path = base_path_counted.joinpath("softmax")

        for step in iterator:
            print(f"In step {step}")
            step = round(step, 2)
            df = pd.read_pickle(dist_path.joinpath("step-{}.pickle".format(step)))
            by_state = df.groupby("original_label")

            for state, frame in by_state:
                sofmax_mean = frame.softmaxed_values.mean()
                softmax_all[state].append(sofmax_mean[state])
                # softmax_means[state].append(sofmax_mean)
        # for keys, values in softmax_means.items():
        #     values = np.asarray(values)
        #     line = prepare_axis(
        #         axis, x_axis, values, SOFMTAX_TITLE.format(config.labels[keys]), X_PERCENT_LABEL, Y_SOFTMAX_LABEL
        #     )
        #     figure.legend(tuple(line), tuple(config.labels))
        #     figure.gca().lines[keys].set_linewidth(4)
        #     figure.savefig(base_path_visualize.joinpath("softmax_class_{}.png".format(config.labels[keys])))
        #     axis.cla()
                
        df = pd.DataFrame(softmax_all)
        df = df.T
        sn.heatmap(df, annot=False, cmap="RdYlGn", ax=axis)
        # axis.set  # annot=True, aby wyświetlać wartości
        axis.set_xticks(np.arange(len(x_axis)) + 0.5, labels=x_axis, rotation=45)
        axis.set_yticklabels([])
        axis.set_title("Mapa cieplna średnich wartości softmax model {}".format(config.model))
        axis.set_xlabel(X_PERCENT_LABEL)
        axis.set_ylabel(Y_SOFTMAX_LABEL)
        figure.savefig(base_path_visualize.joinpath("softmax_{}_class_{}.png".format(config.model, "All")))





def make_average_plots(loaded_config: Config, labels: list, below_rate_indicises: dict | None = None):
    """Tworzy wykresy średnich odległości i liczby sąsiadów w zależności od klasy"""
    def prepare_axis(ax: plt.Axes, x_axis: list, y_axis: np.ndarray,
                     title: str, x_label: str, y_label: str, is_below: np.ndarray | None = None):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid()
        # axes= None
        if is_below is not None:
            # is_below.remove(0)
            below = is_below[is_below != 0]
            for i in range(below.shape[0]):
                ax.plot(
                    x_axis, y_axis, '-D', markevery=np.asanyarray([below[i]])
                )

                if i == below.size - 1:
                    return ax.plot(
                    x_axis, y_axis, '-D', markevery=np.asanyarray([below[i]])
                    )
        return ax.plot(
            x_axis, y_axis,
        )


    for augumentation in loaded_config.augumentations:
        base_path_counted = config.count_base_dir.joinpath(augumentation.template_path)
        if below_rate_indicises is not None:
            below_rate_indicises = below_rate_indicises[augumentation.template_path]
        else:
            below_rate_indicises = None
        base_path_visualize = config.visualization_base_dir.joinpath(
           augumentation.template_path
        )

        base_path_visualize.mkdir(parents=True, exist_ok=True)
        iterator = augumentation.make_iterator()
        mean_distances = {k: {"mahalanobis": [], "cosine": [], "euclidean": [], "neighbours": []} for k in range(10)}
        mean_distance_by_original_class = {k: {"mahalanobis": [], "cosine": [], "euclidean": [], "neighbours": []} for k in range(10)}

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
                mean_mahalanobis = frame.mahalanobis.mean()
                mean_euclidean =  frame.euclidean.mean()
                mean_cosine = frame.cosine.mean()
                mean_neighbours = frame.neighbours.apply(pd.Series).mean().values

                mean_distances[state]['mahalanobis'].append(
                    mean_mahalanobis
                )

                mean_distances[state]['euclidean'].append(
                    mean_euclidean
                )
                mean_distances[state]['cosine'].append(
                    mean_cosine
                )
                mean_distances[state]['neighbours'].append(
                    mean_neighbours
                )
                # Podlicz tak by zobaczyć jak tylko odległości od oryginalnych klas się zmieniają.

                mean_distance_by_original_class[state]['mahalanobis'].append(mean_mahalanobis[state])
                mean_distance_by_original_class[state]['cosine'].append(mean_cosine[state])
                mean_distance_by_original_class[state]['euclidean'].append(mean_euclidean[state])
                mean_distance_by_original_class[state]['neighbours'].append(mean_euclidean[state])


        # values = mean_distance_by_original_class.items()
        mean_neigh_fig, neigh_ax  = plt.subplots( figsize=(10, 10))

        plot_figure, plot_axis = plt.subplots(1, 3, figsize= (18, 10))
        mean_neigh_fig.suptitle("Zmiana liczby sąsiadów dla średniego obrazu z każdej z klas od \n ich oryginalnej przestrzeni z wykorzystaniem augumentacji {}".format(augumentation.name))
        plot_figure.suptitle("Zmiana odległości średniego obrazu z każdej z klas od \n ich oryginalnej przestrzeni z wykorzystaniem augumentacji {}".format(augumentation.name))
        for class_, values in mean_distance_by_original_class.items():
            # print(values)
            plot_axis[0].plot(
            x_axis, values['mahalanobis'], label=labels[class_]
            )
            plot_axis[0].set_title("Mahalanobis")

            plot_axis[1].plot(
            x_axis, values['cosine'], label=labels[class_]
            )
            plot_axis[1].set_title("Cosinusowa")

            plot_axis[2].plot(
            x_axis, values['euclidean'], label=labels[class_]
            )
            plot_axis[2].set_title("Euklidesowa")


            neigh_ax.plot(
            x_axis, values['neighbours'], label=labels[class_]
            )

        handles, labels = plot_figure.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plot_figure.legend(by_label.values(), by_label.keys() ,loc="center right")
        plot_figure.supylabel(X_PERCENT_LABEL)
        plot_figure.supxlabel(X_PERCENT_LABEL)
        mean_neigh_fig.legend()
        neigh_ax.set_xlabel(X_PERCENT_LABEL)
        neigh_ax.set_ylabel("number of neighbours")
        mean_neigh_fig.savefig(distance_visul_path.joinpath("mean-neightbours-all.png"))



        plot_figure.savefig(distance_visul_path.joinpath("mean-distances-all.png"))

        plt.close(plot_figure)
        ### Miejsce gdzie generuję liczbę sąsiadów zmieniającch się w zależności


        # return

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
                              X_PERCENT_LABEL, X_PERCENT_LABEL, below_rate_indicises)
            prepare_axis(plot_ax[1], x_axis, conv_cosine, "cosine", X_PERCENT_LABEL, X_PERCENT_LABEL, below_rate_indicises)
            prepare_axis(plot_ax[2], x_axis, conv_euclid, "euclidean", X_PERCENT_LABEL, X_PERCENT_LABEL, below_rate_indicises)
            plot_fig.legend(
                tuple(l1), tuple(labels),  loc='outside right upper'
            )
            plot_ax[0].grid()
            plot_ax[1].grid()
            plot_ax[2].grid()
            plot_fig.savefig(distance_visul_path.joinpath("mean-distances-{}.png".format(labels[class_])))
            plt.close(plot_fig)
            plot_fig, plot_ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_fig.suptitle(
                "Average number of neighbours for images labeled: {} per percentage\n {} augumentation".format(
                    labels[class_],
                    augumentation.name))
            line = prepare_axis(
                plot_ax, x_axis, conv_neight, "", X_PERCENT_LABEL, NEIGH_LABEL, below_rate_indicises
            )
            plot_ax.grid()
            plot_fig.legend(
                tuple(line), tuple(labels),  loc='outside right upper'
            )
            plot_fig.savefig(neightbours_visul_path.joinpath("mean-neighbours-{}.png".format(labels[class_])))
            plt.close(plot_fig)


if __name__ == "__main__":
    with open("./config-imagenet.json", "r") as file:
        obj = json.load(file)
        models = [SupportedModels(model) for model in obj.get("models")]

        dataset = ImageNetKaggle(root=obj['dataset_path'], split="val", transform=None)

    for tested_model in models:
        obj['model'] = tested_model.value
        config = Config(obj)
        config.dataset = dataset
        prepare_visualization_output_dir(config)
        sn.set_theme()
        # indicise = make_matrix_plots(config, config.labels)
        # print(indicise)
        # make_average_plots(config, config.labels)
        make_average_softmax_plot(config)
    # make_individual_stats(config)
    # generate_step_images(config)

    # import shutil
    # shutil.make_archive("visualizations", "zip", config.visualization_base_dir)