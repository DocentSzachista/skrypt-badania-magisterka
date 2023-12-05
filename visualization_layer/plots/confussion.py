# from ..utils.constants import LABELS_CIFAR_10
# from ..setup import Config
import json
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_confussion_matrix(y_true: list, y_predicted: list, labels: dict,  filename: str, title: str):
    """Generate confussion_matrix heatmap.

        :param y_predicted: predicted classes by neural network
        :type y_predicted: list 1-D of integers in range (0, 9)
        :param y_predicted: labeled classes in test set
        :type y_predicted: list 1-D of integers in range (0, 9)
    """
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=list(labels.keys()))
    print(cf_matrix)
    df_cm = pd.DataFrame(
        cf_matrix, #/ np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in labels.values()],
        columns=[i for i in labels.values()]
    )
    plt.figure(figsize=(10, 10))
    # cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = list(labels.values()))    
    # cm_display.plot()
    plt.title(title)
    # plt.xticks(rotation=45)
    # plt.xlabel("predicted")
    sn.heatmap(
        df_cm, annot=True, fmt="g"
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    print("save")
    plt.savefig(f"{filename}.png")


def make_noise_wide_statistic(y_true: list, y_predicted: list, labels: list, acc_wide: dict, filename: str):
    """Count confussion matrix and bunch of statistics basing on that."""
    cfm = confusion_matrix(y_pred=y_predicted, y_true=y_true, labels=labels)
    for i in labels: 
        TP = cfm[i, i]
        all_samples = np.sum(cfm[i, :])
        acc_wide[i].append( round(TP/all_samples, 4))
    np.save( f"{filename}.npy", cfm)    




# def run(conf: Config, conditions: list, filename: str ):

#     for augumentation in conf.augumentations:
#         path = f"./{conf.model.lower()}-{conf.tag}/{augumentation.name}"
#         files = []
#         # for image_class in LABELS_CIFAR_10.keys():
#         for image_class in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         # image_class = 0
#             files.extend([os.path.join(f"{path}/dataframes/{image_class}", file)
#                      for file in os.listdir(f"{path}/dataframes/{image_class}")   ])
#         dfs = []
#         # for iteration conf.augumentations[0].make_iterator()
#         for file in files:
#             new_df = pd.read_pickle(file)            
#             dfs.append(new_df)
#         df = pd.concat(dfs, ignore_index=True)
#         for v in conditions:
#             print("Make condition")
#             subset = df.loc[df['noise_rate'] == v]
#             y_true = subset['original_label'].values
#             y_pred = subset['predicted_label'].values
#             generate_confussion_matrix(y_true, y_pred, LABELS_CIFAR_10, f"{filename}-{v}", f"confussion matrix for noise: {v} ")