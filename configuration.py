from testing_layer.workflows.augumentations import *
from testing_layer.workflows.enums import *
import pathlib
from testing_layer.workflows.utils import BASE_PATH
from torchvision.datasets import CIFAR10, ImageNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from visualization_layer.constants import *
import os
import gdown

class Config:

    supported_augumentations = {
        SupportedAugumentations.MIXUP: MixupAugumentation,
        SupportedAugumentations.NOISE: NoiseAugumentation
    }

    supported_datasets = {
        SupportedDatasets.CIFAR: CIFAR10,
        SupportedDatasets.IMAGENET: ImageNet
    }

    dataset_labels = {
        SupportedDatasets.CIFAR: LABELS_CIFAR_10,
        SupportedDatasets.IMAGENET: LABELS_IMAGENET
    }

    transform = A.Compose([
        ToTensorV2()
    ])

    def __init__(self, json_config: dict) -> None:
        self.model = json_config.get("model")
        self.tag = json_config.get("tag", "base")
        self.image_dim = json_config.get("image_dim", [3, 32, 32])
        self.augumentations = [
            self.supported_augumentations[SupportedAugumentations(augumentation["name"])](augumentation, self.image_dim)
            for augumentation in json_config.get("augumentations")
        ]
        self.dataset = self.supported_datasets.get(SupportedDatasets(json_config.get("dataset")))(
            root="./datasets", train=False, transforms=lambda x: self.transform(image=np.array(x))["image"].float()/255.0)
        self.labels = self.dataset_labels.get(SupportedDatasets(json_config.get("dataset")))

        self.model_filename = json_config.get("model_location")
        self.g_drive_hash = json_config.get("model_g_drive", None)
        if self.g_drive_hash is not None:
            self.model_filename =f"./{gdown.download(id=self.g_drive_hash)}"


        self.save_preprocessing = json_config.get("save_preprocessing", False)
        self.color_channels = ColorChannels.count_channels(
            json_config.get("chosen_color_chanels", "RGB"))
        self.columns = ["id", "original_label", "predicted_label",
                        "noise_rate", "classifier", "features", "noise_percent"]

        self.count_base_dir = pathlib.Path("./counted_outputs")
        self.visualization_base_dir = pathlib.Path("./visualizations")


def prepare_save_directory(config: Config):
    for augumentation in config.augumentations:
        augumentation.template_path.mkdir(parents=True, exist_ok=True)
        augumentation.template_path.joinpath("dataframes").mkdir(parents=False, exist_ok=True)
        augumentation.template_path.joinpath("images").mkdir(parents=False, exist_ok=True)
    print("Finished creating directories for model outputs")


def prepare_counted_values_output_dir(config: Config):
    for augumentation in config.augumentations:
        path = config.count_base_dir.joinpath(augumentation.template_path)
        # path =   config.count_base_dir.joinpath(BASE_PATH.format(config.model, config.tag, augumentation.name))
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath("matrixes").mkdir(parents=False, exist_ok=True)
        path.joinpath("distances").mkdir(parents=False, exist_ok=True)
    print("Finished creating dirs for counted_output")


def prepare_visualization_output_dir(config: Config):
    for augumentation in config.augumentations:
        path = config.visualization_base_dir.joinpath(
            augumentation.template_path
        )
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath("matrixes").mkdir(parents=False, exist_ok=True)
        path.joinpath("distances").mkdir(parents=False, exist_ok=True)
        path.joinpath("images").mkdir(parents=False, exist_ok=True)
    print("Finished creating dirs for visualization")
