import numpy as np
import copy




class NoiseTransform(object):

    def __init__(self, number_of_pixels: int, shuffled_indexes: list, mask: np.ndarray) -> None:
        self.number_of_pixels = number_of_pixels
        self.shuffled_indexes = shuffled_indexes
        self.mask = mask
        print("Sprawdzanie czy maski sa dobrych rozmiarow")
        print(len(set(self.shuffled_indexes)))
        print(self.mask.shape)
        print(self.number_of_pixels)


    def __call__(self, image: np.ndarray):
        image_copy = copy.deepcopy(image)
        for index in range(self.number_of_pixels):
            i = self.shuffled_indexes[index] // self.mask.shape[1]
            j = self.shuffled_indexes[index] % self.mask.shape[1]
            image_copy[:, i,j] = np.add(image_copy[:, i,j],  self.mask[:, i,j ])
        return image_copy