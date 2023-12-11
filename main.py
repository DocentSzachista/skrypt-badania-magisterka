
from configuration import Config, prepare_save_directory, BASE_PATH
from testing_layer.resnet_cifar_10 import prepare_resnet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
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

def converter(tensor): return tensor.detach().cpu().numpy()


def test_model_with_data_loader(model, data_loader: DataLoader, mask_intensity: int):
        storage = []

        ind = 0
        for _, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
            logits = model(inputs)
            _, predicted = torch.max(logits, dim=1)
            predicted = converter(predicted)
            logits = converter(logits)
            features = get_features(model._modules['1'], inputs)
            for index in range(logits.shape[0]):
                storage.append([
                    ind, converter(targets[index]).item(),
                    predicted[index],
                    mask_intensity,
                    logits[index],
                    converter(features[index]),
                    100*mask_intensity
                ])
                ind+=1

        return storage


def handle_mixup(augumentation: MixupAugumentation, dataset: CIFAR10, iterator: list):
    """Handle mixup augumentation"""
    for class_ in augumentation.classes:
        chosen_indices = [idx for idx, label in enumerate(cifar.targets) if label == class_]

        print("Performing mixup for {}".format(class_))
        for step in iterator: 
            dataset_step = MixupDataset(dataset, chosen_indices, step, path="{}/images/class-{}/{}".format(formatted_path, class_ , step) )
            dataloader =  DataLoader(dataset_step, batch_size=50, shuffle=False, drop_last=False)
            to_save = test_model_with_data_loader(model, dataloader, step)
            df = pd.DataFrame(to_save, columns=conf.columns)
            df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))
            save_path = "{}/dataframes/{}.pickle".format(BASE_PATH.format(conf.model, conf.tag, augumentation.name), step )
            print("Saving...")
            df.to_pickle(save_path)


def handle_noise(augumentation: NoiseAugumentation, dataset: CIFAR10, iterator: list):
    """Handle noise augumentation"""
    for step in iterator: 
        transforms = A.Compose([
            NoiseTransform(number_of_pixels=step, shuffled_indexes=augumentation.shuffled_indexes, mask=augumentation.mask),
            ToTensorV2()
        ])
        dataset.transform = lambda x : transforms(image=np.array(x))["image"].float()/255.0 
        dataloader = DataLoader(cifar, batch_size=50, shuffle=False, drop_last=False)
        to_save = test_model_with_data_loader(model, dataloader, step)
                    
        df = pd.DataFrame(to_save, columns=conf.columns)
        df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))
        save_path = "{}/dataframes/{}.pickle".format(BASE_PATH.format(conf.model, conf.tag, augumentation.name), step )
        print("Saving...")
        df.to_pickle(save_path)



if __name__ == "__main__":
    with open("./config.json", "r") as file:
        obj = json.load(file)
    conf = Config(obj)
    prepare_save_directory(conf)    
    
    model = prepare_resnet(conf.model_filename)
    set_workstation("cuda:0")
    with torch.no_grad():
        model.eval()
        model.to("cuda:0")

        transform = A.Compose([
            ToTensorV2()
        ])
        cifar = CIFAR10("./datasets", train=False,  transform=lambda x: transform(image=np.array(x))["image"].float()/255.0 )
        cat_class_indices = [idx for idx, label in enumerate(cifar.targets) if label == 3]
        
        for augumentation in conf.augumentations: 
            formatted_path = BASE_PATH.format(conf.model, conf.tag, augumentation.name)
            print("current augumentation {}".format(augumentation.name))
            iterator = augumentation.make_iterator()
            if isinstance(augumentation, MixupAugumentation):
                handle_mixup(augumentation, cifar, iterator)
            elif isinstance(augumentation, NoiseAugumentation):
                handle_noise(augumentation, cifar, iterator)
            # TODO: dodawanie transformacji w postaci obrotu obrazka
