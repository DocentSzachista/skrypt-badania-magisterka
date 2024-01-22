import torch.nn as nn
import torch
from .resnet_cifar_10 import change_model_with_normalization
network_features = []


def collect_features(model, input, output):
    global network_features
    network_features.append(input[0].detach().cpu().numpy())
    # network_features.append(input[0])
    # print(network_features)
    # print(len(network_features))


def setup_a_hook(model):
    model.fc.register_forward_hook(collect_features)
    print("Hook initiated")


def prepare_shufflenet():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x2_0", pretrained=True)
    setup_a_hook(model)
    # Przeprowadzenie normalizacji na modelu tak jak w labach ostatnich
    model = change_model_with_normalization(
        model, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    model.cuda()
    model.eval()

    return model
