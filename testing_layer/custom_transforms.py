import numpy as np
import copy
from .workflows.augumentations import generate_mask, create_and_shuffle_indexes



class NoiseTransform(object):

    def __init__(self, number_of_pixels: int, image_shape) -> None:
        self.percentage = number_of_pixels
        self.shape = image_shape
        self.shuffled_indexes = create_and_shuffle_indexes(self.shape)
        self.mask = generate_mask(self.shape)
        self.number_of_pixels = int(self.shape[1]*self.shape[2] * self.percentage)

    def __call__(self, image: np.ndarray):
        image_copy = copy.deepcopy(image)
        for index in range(self.number_of_pixels):
            i = self.shuffled_indexes[index] // self.mask.shape[1]
            j = self.shuffled_indexes[index] % self.mask.shape[1]
            image_copy[:, i,j] = self.mask[:, i,j ]
        return image_copy