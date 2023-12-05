import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.figure as mpf
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from typing import Tuple, Union, List
from ..setup import Config
from ..utils import calculations, constants


def prepare_plotting(figsize: tuple, y_lim: tuple, subplots_dim=(1, 1)) -> Tuple[Union[plt.Axes, List[plt.Axes]], mpf.Figure]:
    sn.set_theme()
    fig, axes = plt.subplots(nrows=subplots_dim[0], ncols=subplots_dim[1], figsize=figsize)
    # axes = fig.add_subplot()
    # plt.ylim(y_lim)
    plt.xticks(rotation=45)
    return axes, fig


def read_image(path: str):
    return mpimg.imread(path)


def _save_gif(figure, anim_func, frames, path, filename):
    ani = FuncAnimation(figure, anim_func, interval=5000, frames=frames,
                        repeat=True, cache_frame_data=False)
    writer = animation.PillowWriter(fps=3, bitrate=1000)
    os.makedirs(path, exist_ok=True)
    ani.save(f"{path}/{filename}", writer=writer)


def prepare_distance_data(distance_func: calculations.Distance, dataset: pd.DataFrame, column_name: str):

    from_origin_dist = []
    from_fully_augumented_dist = []

    values = dataset[column_name].to_numpy()
    origin = values[0]
    final_conversion = values[-1]

    for index in range(len(values)):
        from_origin_dist.append(
            distance_func.count_distance(origin, values[index])
        )
        from_fully_augumented_dist.append(
            distance_func.count_distance(final_conversion, values[index])
        )
        # print(from_origin_dist)
    return from_origin_dist, from_fully_augumented_dist


def prepare_distance_data_mean(
        distance_func: calculations.Distance, train_dataset: pd.DataFrame, research_features: pd.DataFrame,
        column_name: str):
    """Prepares distances data from each class."""
    res_points = research_features[column_name]

    # prepare to lookout
    subsets_train = {k: train_dataset.loc[train_dataset["original_label"] == k]
                     [column_name].to_numpy() for k in range(0, 10)}

    actual_distances = {
        k: [] for k in range(0, 10)
    }

    for i in range(0, 10):
        subset_np = subsets_train.get(i)
        for point in res_points:
            distances = [distance_func.count_distance(point, point_train) for point_train in subset_np]
            mean_dist = np.mean(distances)
            actual_distances[i].append(mean_dist)

    return list(actual_distances.values())


def make_bar_plot(
    df: pd.DataFrame, img_id: str, save_path: str, datas_heights: tuple,
    x_labels: list, y_lim: tuple, class_name: str, file_name: str, path_to_images=None
):

    if path_to_images is not None:
        axes, fig = prepare_plotting((8, 7), y_lim, (1, 2))
        axes[0].set_ylim(top=y_lim[1], bottom=y_lim[0])
        axes[0].set_aspect(aspect="auto")
        bar_plot = axes[0].bar(
            x_labels, [0]*len(x_labels), align="center",
        )
        image = read_image(f"{path_to_images}/images/image{img_id}_4_noise_{df['noise_rate'][0]}.png")
        axes[1].set_xlim(right=32)
        axes[1].set_ylim(top=32)
        image_plot = axes[1].imshow(
            image,
        )
    else:
        axes, fig = prepare_plotting((8, 7), y_lim)
        bar_plot = axes.bar(x_labels, [0]*len(x_labels), align="center")
        axes.set_ylim(top=y_lim[1], bottom=y_lim[0])

    def animate(index):

        title = "id: {class_name}_{id} noise value: {pixels}, noise %: {percentage}".format(
            class_name=class_name, id=img_id, pixels=df['noise_rate'][index], percentage=df['noise_percent'][index])
        fig.suptitle(title, fontsize=8)
        # axes.set_title()
        if path_to_images:
            image = read_image(f"{path_to_images}/images/image{img_id}_4_noise_{df['noise_rate'][index]}.png")
            image_plot.set_data(image)
        # print(datas_heights)
        for j in range(len(datas_heights)):
            bar_plot[j].set_height(datas_heights[j][index])

    _save_gif(fig, animate, len(datas_heights[0]) - 1,
              f"{save_path}{class_name}/{img_id}",
              f"{file_name}.gif")
    plt.close()


def make_plot(x_points: list, y_points: tuple, y_lim: tuple, title: str, save_path: str, filename: str, legend: list):
    axes, fig = prepare_plotting((8, 7), y_lim)

    for index in range(len(y_points)):
        axes.plot(x_points, y_points[index], label=legend[index])
    axes.set_title(title)
    axes.set_xlabel('noise rate')
    axes.set_ylabel('distance')
    fig.legend()
    fig.savefig(f"{save_path}/{filename}.png")


def run(config_file_path="./config.json"):
    """Runs gifs generation basing on a config
       file used to test neural network.
    """

    dist_funcs = [
        calculations.CosineDistance(),
        calculations.EuclidianDistance()
    ]

    with open(config_file_path, "r") as file:
        config = json.load(file)
        conf = Config(config)

        malanobis = calculations.MahalanobisDistance()
        malanobis.fit(conf.training_df)
        dist_funcs.append(malanobis)




        for augumentation in conf.augumentations:
            path = f"./{conf.model.name.lower()}-{conf.tag}/{augumentation.name}"
            files = [os.path.join(f"{path}/dataframes", file)
                     for file in os.listdir(f"{path}/dataframes")]
            os.makedirs(f"./out/{path}", exist_ok=True)
            for file in files:
                df = pd.read_pickle(file)
                img_id = re.findall(r"\_\d+", file)[0]
                class_name = constants.LABELS_CIFAR_10.get(df['original_label'][0], "Error")
                logits = df.classifier.to_numpy()
                logits_tuple = np.array([np.squeeze(row, axis=0) for row in logits]).T

                make_bar_plot(
                    df, img_id, f"./out/{path}/",
                    logits_tuple, list(constants.LABELS_CIFAR_10.values()),
                    (-20, 20), class_name, "logits")

                # math_scores = {k.name: [] for k in calculations.DISTANCE_FUNCS}

                for func in dist_funcs:


                    distances = prepare_distance_data(func, df, "features")
                    make_bar_plot(
                        df, img_id, f"./out/{path}/", distances,
                        ["origin", "augumented"], func.y_lim, class_name,
                        func.name, path_to_images=path
                    )
                    make_plot(y_points=distances, x_points=df.noise_percent.to_numpy(),
                              y_lim=func.y_lim, title=f"{class_name} {func.name} distance",
                              save_path=f"./out/{path}/{class_name}/{img_id}", filename=f"{func.name}_line_plot",
                              legend=["original", "fully transformed"]
                              )


                    if not isinstance(func, calculations.MahalanobisDistance):
                        mean_distances = prepare_distance_data_mean(func, conf.training_df, df, "features")
                        make_bar_plot(df, img_id, f"./out/{path}/", mean_distances,
                                    list(constants.LABELS_CIFAR_10.values()),
                                    func.y_lim, class_name, f"{func.name}_mean")
                        make_plot(y_points=mean_distances, x_points=df.noise_percent.to_numpy(),
                                y_lim=func.y_lim, title=f"{class_name} {func.name} mean distance",
                                save_path=f"./out/{path}/{class_name}/{img_id}", filename=f"{func.name}_mean_line_plot",
                                legend=list(constants.LABELS_CIFAR_10.values()))

