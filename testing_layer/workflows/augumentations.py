import numpy as np
import torch
from .utils import BASE_PATH
import pathlib
import copy
import torchvision

def create_and_shuffle_indexes(matrix_shape: tuple):
    np.random.seed(0)
    indexes = [
        i * matrix_shape[1] + j for i in range(matrix_shape[1]) for j in range(matrix_shape[1])
    ]
    np.random.shuffle(indexes)
    return indexes


def generate_mask(shape: tuple):
    temp = torch.rand(shape)
    # Wszystkie modele mają taką samą normalizację
    temp = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(temp)
    return temp


def apply_noise(image: torch.Tensor, mask: np.ndarray, number_of_pixels: int, shuffled_indexes: list):
        image_copy = copy.deepcopy(image)
        image_length = image.shape[1]
        for index in range(number_of_pixels):
            i = shuffled_indexes[index] // image_length
            j = shuffled_indexes[index] % image_length
            image_copy[:,i, j] = np.add(image_copy[:,i, j],  mask[:,i, j])
        return image_copy



class BaseAugumentation:
    def __init__(self, config: dict, tag: str, model_name: str) -> None:
        self.tag = tag
        self.model_name =model_name
        self.name = config.get("name")
        self.start_point = config.get("start_point")
        self.finish_point = config.get("finish_point")
        self.step = config.get("step")
        self.template_path : pathlib.Path
    def make_iterator(self):
        step = int((self.finish_point - self.start_point) / self.step)+1
        return np.linspace(self.start_point, self.finish_point, step, endpoint=True)


class NoiseAugumentation(BaseAugumentation):
    def __init__(self, config: dict, image_dim: list, tag: str, model_name: str) -> None:
        super().__init__(config, tag, model_name)
        self.max_size = image_dim[1] * image_dim[2]
        self.template_path = pathlib.Path(
            BASE_PATH.format(self.model_name, self.tag, self.name))
        self.image_size = (3, 32, 32)


class MixupAugumentation(BaseAugumentation):
    def __init__(self, config: dict, image_dim: list, tag: str, model_name: str) -> None:
        super().__init__(config, tag, model_name)
        self.max_size = 100
        self.class_ = config["picked_class"]
        self.template_path =  pathlib.Path(
            BASE_PATH.format(self.model_name, self.tag, f"{self.name}-{self.class_}"))
