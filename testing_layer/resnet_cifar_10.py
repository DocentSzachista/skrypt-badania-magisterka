import numpy as np
import torch
import torchvision


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


def prepare_resnet( resnet_location: str, num_classes=10) -> torchvision.models.ResNet:
    model = torchvision.models.resnet152(num_classes=num_classes)

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
