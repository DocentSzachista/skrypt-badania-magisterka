from enum import Enum


class SupportedModels(Enum):
    RESNET = "resnet"


class ColorChannels(Enum):
    R = 0
    G = 1
    B = 2

    @staticmethod
    def count_channels(channels: str):
        try:
            return [ColorChannels[channel].value for channel in channels]
        except KeyError:
            raise ValueError("Supplied incorrect channel name")


class SupportedAugumentations(Enum):
    NOISE = "noise"
    MIXUP = "mixup"


class SupportedDatasets(Enum):
    CIFAR = "cifar_10"
