from testing_layer.workflows.augumentations import *
from testing_layer.workflows.enums import *
import pathlib
from torchvision.datasets import CIFAR10, ImageNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from visualization_layer.constants import *
from .datasets import ImageNetKaggle

import os

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
        self.num_workers = json_config['num_workers']
        self.should_override = json_config['override_existing']
        self.device = json_config['device']
        self.model = json_config['model']
        self.batch_size = json_config['batch_size']
        self.tag = json_config.get("tag", "base")
        self.chosen_train_set = "./cifar_10.pickle" if self.model == 'resnet' else "./datasets/shufflenet_train_set.pickle"
        self.image_dim = json_config.get("image_dim", [3, 32, 32])
        self.labels = self.dataset_labels.get(SupportedDatasets(json_config.get("dataset")))
        self.augumentations = [
            self.supported_augumentations[SupportedAugumentations(augumentation["name"])]
            (augumentation, self.image_dim, tag=self.tag, model_name=self.model)
            for augumentation in json_config.get("augumentations")]
        template_mixup = self.augumentations.pop()
        if isinstance(template_mixup, MixupAugumentation):
            for label in range(len(self.labels)):
                temp = copy.deepcopy(template_mixup)
                temp.class_ = label
                self.augumentations.append(temp)
        else:
            # obviously not a mixup
            self.augumentations = [template_mixup]

        # self.dataset = ImageNetKaggle(root=json_config['dataset_path'], split="val", transform=lambda x: self.transform(image=np.array(x))["image"].float()/255.0)

        # self.dataset = self.supported_datasets.get(SupportedDatasets(json_config.get("dataset")))(
        #     root=json_config['dataset_path'], split="val", train=False, download=True, transform=lambda x: self.transform(image=np.array(x))["image"].float()/255.0)

        self.model_filename = json_config.get("model_location", None)
        self.g_drive_hash = json_config.get("model_g_drive", None)

        # if self.g_drive_hash is not None and not os.path.isfile(self.model_filename):
        #     self.model_filename = f"./{gdown.download(id=self.g_drive_hash)}"

        self.save_preprocessing = json_config.get("save_preprocessing", False)
        self.color_channels = ColorChannels.count_channels(
            json_config.get("chosen_color_chanels", "RGB"))
        self.columns = ["id", "original_label", "predicted_label",
                        "noise_rate", "classifier", "features", "noise_percent"]
        self.model_base_dir = pathlib.Path(f"./models_script_output/{self.model}-{self.tag}")
        self.count_base_dir = pathlib.Path("./counted_outputs")
        self.visualization_base_dir = pathlib.Path("./visualizations")


def prepare_save_directory(config: Config):
    for augumentation in config.augumentations:
        augumentation.template_path.mkdir(parents=True, exist_ok=True)
        augumentation.template_path.joinpath("dataframes").mkdir(parents=False, exist_ok=True)
        # augumentation.template_path.joinpath("images").mkdir(parents=False, exist_ok=True)
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
