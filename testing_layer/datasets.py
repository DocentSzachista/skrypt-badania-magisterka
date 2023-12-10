from torch.utils.data import Dataset
import numpy as np 
from torchvision.utils import save_image
from os import makedirs
DEBUG_SAVE_PATH = "./example"
from pathlib import Path

class MixupDataset(Dataset):

    def __init__(self, dataset, chosen_class_indices, alpha=0.2, should_save_processing=True, path=Path()):
        self.dataset = dataset
        self.chosen_class_indices = chosen_class_indices
        self.alpha = alpha
        self.shoud_save = should_save_processing
        self.path = path
        makedirs(self.path, exist_ok= True)
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
            path.mkdir(parents=True, exist_ok=True)
            save_image(mixed_img, path.joinpath(f"{label}-id-{index}-to-{chosen_label}-id-{non_object}-step-{self.alpha}.jpeg"))
        
        return mixed_img, label


# class NoiseDataset(Dataset): 
    
#     def __init__(self, dataset, mask, indexes: list, number_of_pixels: int, should_save_processing=True):
#         self.dataset = dataset
#         self.mask = mask
#         self.indexes = indexes
#         self.number_of_pixels = number_of_pixels
    
#     def __len__(self):
#         return len(self.dataset)
    

#     def __getitem__(self, index):
#         img, label = self.dataset[index]
#         # Losowy indeks obrazu z innej klasy (nie kot)
#         non_object = np.random.choice([
#             idx for idx in range(len(self.dataset)) if idx != label and idx in self.chosen_class_indices
#         ])
#         chosen_image, chosen_label = self.dataset[non_object]
#         # Mixup - łączenie obrazów z etykietami
#         mixed_img = self.alpha * chosen_image + (1 - self.alpha) * img 
#         if self.shoud_save:
#             save_image(mixed_img, f"{self.path}/{label}-id-{index}-to-{chosen_label}-id-{non_object}.jpeg" )
        
#         return mixed_img, label