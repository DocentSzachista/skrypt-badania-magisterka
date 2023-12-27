import torch
from .workflows.enums import *
from .configuration import Config
from torchvision.models import resnet152, ResNet152_Weights, ResNet

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer("mean", torch.Tensor(mean))
        self.register_buffer("std", torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def change_model_with_normalization(model, mean, std):
    norm_layer = Normalize(mean=mean, std=std)
    return torch.nn.Sequential(norm_layer, model)


def prepare_resnet( resnet_location: str, num_classes=10) -> ResNet:
    model = resnet152(num_classes=num_classes)

    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()

    state_dict = torch.load(resnet_location, map_location="cuda:0")[
        "state_dict"
    ]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    # Przeprowadzenie normalizacji na modelu tak jak w labach ostatnich
    model = change_model_with_normalization(
        model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    model.cuda()
    model.eval()

    return model



def load_resnet_model(config: Config):
    if config.dataset == SupportedDatasets.CIFAR:
        model = resnet152(num_classes=10)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()

        state_dict = torch.load(config.model_filename, map_location="cuda:0")[
            "state_dict"
        ]
        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        # Przeprowadzenie normalizacji na modelu tak jak w labach ostatnich
        model = change_model_with_normalization(
            model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
        # model.cuda()
        # model.eval()
    elif config.dataset == SupportedDatasets.IMAGENET:
        model = resnet152(num_classes=1000, weights=ResNet152_Weights.IMAGENET1K_V1)
        model = change_model_with_normalization(
            model, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
    return model
