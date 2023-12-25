from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from torchvision.utils import save_image
from os import makedirs


class MixupDataset(Dataset):

    def __init__(self, dataset, chosen_class_indices, alpha=0.2, should_save_processing=False, path=Path()):
        self.dataset = dataset
        self.chosen_class_indices = chosen_class_indices
        self.alpha = alpha
        self.shoud_save = should_save_processing
        self.path = path
        np.random.seed(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Losowy indeks obrazu z wybranego indeksu
        non_object = np.random.choice([
            idx for idx in range(len(self.dataset)) if idx != label and idx in self.chosen_class_indices
        ])
        chosen_image, chosen_label = self.dataset[non_object]
        # Mixup - łączenie obrazów z etykietami
        mixed_img = self.alpha * chosen_image + (1 - self.alpha) * img
        if self.shoud_save and self.path:
            path = self.path.joinpath(f"{label}/{index}")
            makedirs(self.path, exist_ok=True)
            path.mkdir(parents=True, exist_ok=True)
            save_image(mixed_img, path.joinpath(
                f"{label}-id-{index}-to-{chosen_label}-id-{non_object}-step-{self.alpha}.jpeg"))

        return mixed_img, label
