from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from torchvision.utils import save_image
import os
from PIL import Image
import json

class MixupDataset(Dataset):

    def __init__(self, dataset, chosen_class_indices, alpha=0.2, should_save_processing=False, path=Path()):
        self.dataset = dataset
        self.transform = dataset.transform
        self.chosen_class_indices = chosen_class_indices
        self.alpha = alpha
        self.shoud_save = should_save_processing
        self.path = path
        np.random.seed(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):


        img, label = self.dataset[index]
        img = self.transform(img)
        # Losowy indeks obrazu z wybranego indeksu
        non_object = np.random.choice([
            idx for idx in range(len(self.dataset)) if idx != label and idx in self.chosen_class_indices
        ])
        chosen_image, chosen_label = self.dataset[non_object]
        chosen_image = self.transform(chosen_image)
        # Mixup - łączenie obrazów z etykietami
        mixed_img = self.alpha * chosen_image + (1 - self.alpha) * img
        if self.shoud_save and self.path:
            path = self.path.joinpath(f"{label}/{index}")
            os.makedirs(self.path, exist_ok=True)
            path.mkdir(parents=True, exist_ok=True)
            save_image(mixed_img, path.joinpath(
                f"{label}-id-{index}-to-{chosen_label}-id-{non_object}-step-{self.alpha}.jpeg"))

        return mixed_img, label




class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None, class_id = -1):
        self.class_id = class_id
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
