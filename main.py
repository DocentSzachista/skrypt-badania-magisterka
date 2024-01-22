
from testing_layer.configuration import Config, prepare_save_directory
from testing_layer.resnet_cifar_10 import prepare_resnet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10, ImageNet
from testing_layer.workflows.augumentations import *
import numpy as np
from testing_layer.datasets import MixupDataset
from torch.utils.data import DataLoader
from testing_layer.workflows.utils import set_workstation
import torch
import pandas as pd
from testing_layer.workflows.cifar_10 import get_features
import json
from testing_layer.custom_transforms import NoiseTransform
from testing_layer.shufflenet import prepare_shufflenet, network_features
from torchvision.models import ResNet
def converter(tensor): return tensor.detach().cpu().numpy()


def test_model_with_data_loader(model, data_loader: DataLoader, mask_intensity: int):
    global network_features
    storage = []

    ind = 0
    for batch, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
        logits = model(inputs)
        _, predicted = torch.max(logits, dim=1)
        predicted = converter(predicted)
        logits = converter(logits)
        if isinstance(model, ResNet):
            features = converter(get_features(model._modules['1'], inputs))
        else:
            features = network_features[batch]
            # print("Pokazuj batcha")
            # print(len(features.shape))
            # print(logits.shape[0])
        for index in range(logits.shape[0]):
            storage.append([
                ind, converter(targets[index]).item(),
                predicted[index],
                mask_intensity,
                logits[index],
                features[index],
                100*mask_intensity
            ])
            ind += 1
        # if not isinstance(model, ResNet):
        #     network_features.clear()
        #     print("czyść")
        #     print(len(network_features))
    # print(len(network_features))
    network_features.clear()
    return storage


def handle_mixup(augumentation: MixupAugumentation, dataset: CIFAR10 | ImageNet, iterator: list):
    """Handle mixup augumentation"""
    chosen_indices = [idx for idx, label in enumerate(dataset.targets) if label == augumentation.class_]
    print("Performing mixup for {}".format(augumentation.class_))
    for step in iterator:
        dataset_step = MixupDataset(
            dataset, chosen_indices, step, path="{}/images/class-{}/{}".format(
                augumentation.template_path, augumentation.class_, step),
            should_save_processing=False)
        dataloader = DataLoader(dataset_step, batch_size=50, shuffle=False, drop_last=False)
        to_save = test_model_with_data_loader(model, dataloader, step)
        df = pd.DataFrame(to_save, columns=conf.columns)
        df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))

        save_path = "{}/dataframes/{}.pickle".format(augumentation.template_path, round(step, 2))
        print("Saving...")
        df.to_pickle(save_path)


def handle_noise(augumentation: NoiseAugumentation, dataset: CIFAR10 | ImageNet, iterator: list):
    """Handle noise augumentation"""
    for step in iterator:
        transforms = A.Compose([
            NoiseTransform(
                number_of_pixels=step, shuffled_indexes=augumentation.shuffled_indexes,
                mask=augumentation.mask),
            ToTensorV2()])
        dataset.transform = lambda x: transforms(image=np.array(x))["image"].float()/255.0
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False, drop_last=False)
        to_save = test_model_with_data_loader(model, dataloader, step)

        df = pd.DataFrame(to_save, columns=conf.columns)
        df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))
        save_path = "{}/dataframes/{}.pickle".format(augumentation.template_path, round(step, 2))
        print("Saving...")
        df.to_pickle(save_path)


if __name__ == "__main__":
    with open("./config-mixup.json", "r") as file:
        obj = json.load(file)
    conf = Config(obj)
    prepare_save_directory(conf)
    print(torch.cuda.is_available())
    if conf.model == "resnet":
        model = prepare_resnet(conf.model_filename)
    else:
        model = prepare_shufflenet()
    set_workstation("cuda:0")
    with torch.no_grad():
        model.eval()
        model.cuda()

        for augumentation in conf.augumentations:
            formatted_path = augumentation.template_path
            print("current augumentation {}".format(augumentation.name))
            iterator = augumentation.make_iterator()
            if isinstance(augumentation, MixupAugumentation):
                handle_mixup(augumentation, conf.dataset, iterator)
            elif isinstance(augumentation, NoiseAugumentation):
                handle_noise(augumentation, conf.dataset, iterator)


    # import shutil
    # shutil.make_archive("model", "zip", conf.model_base_dir)
