
import pandas as pd
import torch
import torchvision
from albumentations.pytorch import ToTensorV2

from .utils import set_workstation

columns = [
    "id",
    "original_label",
    "predicted_label",
    "noise_rate",
    "classifier",
    "features",
]
old_columns = ["id", "original_label", "classifier", "features"]
converter = lambda tensor: tensor.detach().cpu().numpy()


def save_to_pickle_file(data: list, columns: list, filepath: str):
    """Save collected data about model performance to pickle file"""
    df = pd.DataFrame(data, columns=columns)
    df.to_pickle(filepath)
    return df


def get_features(model, data):
    """Retrieves feature values from avgpool layer of network."""
    for name, module in model._modules.items():
        data = module(data)
        # print(name, data.shape)
        if name == "avgpool":
            # tu są cechy które chcę wyciągnąć z sieci
            data = data.view(-1, 2048)
            return data


def retrieve_images_from_dataloader(preprocess, image_ids: list):
    test_set = torchvision.datasets.CIFAR10(
        root="./", train=False, download=True, transform=ToTensorV2()
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    test_dataset = test_loader.dataset
    images = [test_dataset[i][0] for i in image_ids]
    labels = [test_dataset[i][1] for i in image_ids]

    return labels, images


def retrieve_chosen_images_ids(filepath: str):
    chosen = pd.read_pickle(filepath)
    return chosen["id"].to_numpy()
