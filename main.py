
from testing_layer.configuration import Config, prepare_save_directory
from testing_layer.resnet_cifar_10 import prepare_resnet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10, ImageNet
from testing_layer.workflows.augumentations import *
from testing_layer.datasets import MixupDataset, ImageNetKaggle, MixupNoiseDataset
from torch.utils.data import DataLoader
from testing_layer.workflows.utils import set_workstation
import torch
import pandas as pd
from testing_layer.workflows.cifar_10 import get_features
import json
from testing_layer.custom_transforms import NoiseTransform
from testing_layer.model_loading import *
from testing_layer.model_loading import load_model
from testing_layer.workflows.enums import SupportedModels
from testing_layer.workflows.augumentations import apply_noise
import torchvision
import torchvision.transforms as transforms
import os


def converter(tensor): return tensor.detach().cpu().numpy()


def test_model_with_data_loader(model, data_loader: DataLoader, mask_intensity: int, augumentation: BaseAugumentation, device: str):
    global network_features
    storage = []

    ind = 0
    for batch, (inputs, targets) in enumerate(data_loader):
        print("Testing batch {}/{}".format(batch, len(data_loader)))
        # if isinstance(augumentation, NoiseAugumentation):
        #     new_inputs = []
        #     for image in inputs:
        #         new_inputs.append(
        #             apply_noise(image, augumentation.mask, mask_intensity, augumentation.shuffled_indexes, image.shape[2] )
        #         )
        #     inputs = torch.stack(new_inputs)
        # print(f"Shape: {inputs[0].shape}")
        torchvision.utils.save_image(inputs[0], f"./porownywarka/Testowy_obraz_{batch}.png")
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        _, predicted = torch.max(logits, dim=1)
        predicted = converter(predicted)
        logits = converter(logits)
        features = network_features[batch]
        # print("Pokazuj batcha")
        # print(features.shape)
        # print(logits.shape[0])
        for index in range(logits.shape[0]):
            # print(predicted[index], converter(targets[index]).item() )
            storage.append([
                ind, converter(targets[index]).item(),
                predicted[index],
                mask_intensity,
                logits[index],
                features[index],
                100*mask_intensity
            ])
            ind += 1

    # network_features.clear()
    # return storage


def handle_mixup(augumentation: MixupAugumentation, dataset: ImageNetKaggle, iterator: list, batch_size : int, device: str):
    """Handle mixup augumentation"""
    chosen_indices = [idx for idx, label in enumerate(dataset.targets) if label == augumentation.class_]
    # dataset.chosen_class_indices = chosen_indices
    print("Performing mixup for {}".format(augumentation.class_))
    for step in iterator:
        dataset_step = MixupDataset(
            dataset, chosen_indices, step, path="{}/images/class-{}/{}".format(
                augumentation.template_path, augumentation.class_, step),
            should_save_processing=False)
        # dataset.alpha = step
        dataloader = DataLoader(dataset_step, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)
        to_save = test_model_with_data_loader(model, dataloader, step, augumentation, device)
        df = pd.DataFrame(to_save, columns=conf.columns)
        df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))

        save_path = "{}/dataframes/{}.pickle".format(augumentation.template_path, round(step, 2))
        print("Saving...")
        df.to_pickle(save_path)

def handle_noise_mixup(augumentation: MixupNoiseAugumentation, dataset: ImageNetKaggle, iterator: list, batch_size : int, device: str, num_workers: int):
    """Handle mixup augumentation"""
    chosen_indices = [idx for idx, label in enumerate(dataset.targets) if label == augumentation.class_]
    print("Performing noisemixup for {}".format(augumentation.class_))
    for step in iterator:
        dataset_step = MixupDataset(
            dataset, chosen_indices, step),
        # dataset.alpha = step
        dataloader = DataLoader(dataset_step, batch_size=batch_size, shuffle=False, drop_last=False)
        to_save = test_model_with_data_loader(model, dataloader, step, augumentation, device)

        df = pd.DataFrame(to_save, columns=conf.columns)
        df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))

        save_path = "{}/dataframes/{}.pickle".format(augumentation.template_path, round(step, 2))
        print("Saving...")
        df.to_pickle(save_path)




def handle_noise(transformations, augumentation: NoiseAugumentation, dataset: CIFAR10 | ImageNet, iterator: list, batch_size: int, num_workers: int, device: str, should_override: bool):
    """Handle noise augumentation"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    for step in iterator:
        save_path = "{}/dataframes/{}.pickle".format(augumentation.template_path, round(step, 2))
        perc = round( step*100 /augumentation.max_size, 2)
        if os.path.isfile(save_path) and not should_override:
            print("File detected: skip testing for {}".format(perc))
            continue
        print("Testing model on {}% pixels of image being pertubated with random noise.".format(perc))
        transformation = transforms.Compose([
            transformations,
            NoiseTransform(
                number_of_pixels=step, shuffled_indexes=augumentation.shuffled_indexes,
                mask=augumentation.mask)]
        )
        dataset.transform = transformation
        to_save = test_model_with_data_loader(model, dataloader, step, augumentation, device)
        df = pd.DataFrame(to_save, columns=conf.columns)
        df["noise_percent"] = df["noise_percent"].apply(lambda numb: round(numb / augumentation.max_size, 2))
        # save_path = "{}/dataframes/{}.pickle".format(augumentation.template_path, round(step, 2))

        print(f"Saving results for {perc} % pixels of image being pertubated with random noise")
        df.to_pickle(save_path)


if __name__ == "__main__":
    with open("./config-imagenet.json", "r") as file:
        obj = json.load(file)
        set_workstation(obj['device'])
        models = [ SupportedModels(model) for model in obj.get("models")]
    for tested_model in models:
        obj['model'] = tested_model.value
        conf = Config(obj)
        prepare_save_directory(conf)
        print("Loading model: {}".format(tested_model.value))
        model, transformations = load_model(tested_model)
        dataset = ImageNetKaggle(root=obj['dataset_path'], split="val", transform=transformations)

        print("Setting hooks for the model")
        hook = setup_a_hook(model, tested_model)
        # print(transformations)
        # break
        with torch.no_grad():
            model.eval()
            model.to(conf.device)
            for augumentation in conf.augumentations:
                if(isinstance(augumentation,NoiseAugumentation)):
                    new_size = transformations.crop_size[0]
                    augumentation.generate_new_mask((3, new_size, new_size))
                elif isinstance(augumentation, MixupNoiseAugumentation):
                    new_size = transformations.crop_size[0]
                    augumentation.set_indexes((3, new_size, new_size))
                formatted_path = augumentation.template_path
                print("current augumentation {}".format(augumentation.name))
                iterator = augumentation.make_iterator()
                if isinstance(augumentation, MixupAugumentation):
                    handle_mixup(augumentation, dataset, iterator, conf.batch_size, conf.device)
                elif isinstance(augumentation, NoiseAugumentation):
                    handle_noise(transformations, augumentation, dataset, iterator, conf.batch_size, conf.num_workers, conf.device, conf.should_override)
                elif isinstance(augumentation, MixupNoiseAugumentation):
                    handle_noise_mixup(augumentation, dataset, iterator, conf.batch_size,  conf.device, conf.num_workers)

        print("Remove hooks")
        remove_hook(hook)


    # import shutil
    # shutil.make_archive("model", "zip", conf.model_base_dir)
