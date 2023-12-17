from testing_layer.workflows.augumentations import *
from testing_layer.workflows.enums import *
from visualization_layer.constants import *
import pathlib

BASE_PATH = "{}-{}/{}"

class Config:

    supported_augumentations = {
        SupportedAugumentations.MIXUP: MixupAugumentation,
        SupportedAugumentations.NOISE: NoiseAugumentation
    }

    supported_datasets = {
        SupportedDatasets.CIFAR: LABELS_CIFAR_10,
        SupportedDatasets.IMAGENET: LABELS_IMAGENET
    }

    def __init__(self, json_config: dict) -> None:
        self.model = json_config.get("model")
        self.tag = json_config.get("tag", "base")
        self.image_dim = json_config.get("image_dim", [3, 32, 32])
        self.augumentations = [
            self.supported_augumentations.get(SupportedAugumentations(augumentation["name"]))(augumentation, self.image_dim)
            for augumentation in json_config.get("augumentations")
        ]
        chosen_dataset = SupportedDatasets(json_config.get("dataset"))
        # self.dataset = TODO: Dodanie pobierania warunkowego w zależności od tej zmiennej

        self.model_filename = json_config.get("model_location")
        self.g_drive_hash = json_config.get("model_g_drive")
        self.save_preprocessing = json_config.get("save_preprocessing", False)
        self.color_channels = ColorChannels.count_channels(
            json_config.get("chosen_color_chanels", "RGB"))
        self.columns = ["id", "original_label", "predicted_label",
                        "noise_rate", "classifier", "features", "noise_percent"]

        self.count_base_dir = pathlib.Path("./counted_outputs")
        self.visualization_base_dir = pathlib.Path("./visualizations")
        self.dataset_labels = self.supported_datasets.get(chosen_dataset)


def prepare_save_directory(config : Config):
    for augumentation in config.augumentations:
        path = pathlib.Path(
            BASE_PATH.format(config.model, config.tag, augumentation.name))
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath("dataframes").mkdir(parents=False, exist_ok=True)
        path.joinpath("images").mkdir(parents=False, exist_ok=True)
    print("Finished creating directories for model outputs")


def prepare_counted_values_output_dir(config: Config):
    for augumentation in config.augumentations:
        if isinstance(augumentation, MixupAugumentation):
            for class_ in augumentation.classes:
                path = config.count_base_dir.joinpath(BASE_PATH.format(config.model, config.tag, augumentation.name).join("mixup_to_{}").format(class_))
                path.mkdir(parents=True, exist_ok=True)
                path.joinpath("matrixes").mkdir(parents=False, exist_ok=True)
                path.joinpath("distances").mkdir(parents=False, exist_ok=True)
        else:
            path = config.count_base_dir.joinpath(BASE_PATH.format(config.model, config.tag, augumentation.name))
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath("matrixes").mkdir(parents=False, exist_ok=True)
            path.joinpath("distances").mkdir(parents=False, exist_ok=True)
    print("Finished creating dirs for counted_output")


def prepare_visualization_output_dir(config: Config):
    for augumentation in config.augumentations:
        if isinstance(augumentation, MixupAugumentation):
            for class_ in augumentation.classes:
                path = config.visualization_base_dir.joinpath(BASE_PATH.format(config.model, config.tag, augumentation.name).join("mixup_to_{}").format(class_))
                path.mkdir(parents=True, exist_ok=True)
                path.joinpath("matrixes").mkdir(parents=False, exist_ok=True)
                path.joinpath("distances").mkdir(parents=False, exist_ok=True)
                path.joinpath("images").mkdir(parents=False, exist_ok=True)
        else:
            path = config.visualization_base_dir.joinpath(BASE_PATH.format(config.model, config.tag, augumentation.name))
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath("matrixes").mkdir(parents=False, exist_ok=True)
            path.joinpath("distances").mkdir(parents=False, exist_ok=True)
            path.joinpath("images").mkdir(parents=False, exist_ok=True)
    print("Finished creating dirs for visualization")
