from testing_layer.shufflenet import prepare_shufflenet, network_features
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from testing_layer.workflows.utils import set_workstation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from main import converter
import pandas as pd
import numpy as np
from torchvision.models import resnet152

if __name__ == "__main__":
    model = resnet152(num_classes=10)
    print(model)
    # transform =     transform = A.Compose([
    #     ToTensorV2()
    # ])
    # model = prepare_shufflenet()
    # model.cuda()
    # dataset = CIFAR10(
    #         root="./datasets", train=True, download=True, transform=lambda x: transform(image=np.array(x))["image"].float()/255.0)
    # data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
    # storage = []
    # ind = 0
    # for batch, (inputs, targets) in enumerate(data_loader):
    #     inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
    #     logits = model(inputs)
    #     _, predicted = torch.max(logits, dim=1)
    #     predicted = converter(predicted)
    #     logits = converter(logits)
    #     features = network_features[batch]
    #     for index in range(logits.shape[0]):
    #         storage.append([
    #             ind, converter(targets[index]).item(),
    #             predicted[index],
    #             logits[index],
    #             features[index],
    #         ])
    #         ind += 1
    # columns = ["id", "original_label", "predicted_label",
    #                     "classifier", "features", ]
    # df = pd.DataFrame(storage, columns=columns)
    # df.to_pickle("./datasets/shufflenet_train_set.pickle")
