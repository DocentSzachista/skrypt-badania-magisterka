import numpy as np
import copy


class NoiseTransform(object):

    def __init__(self, number_of_pixels: int, shuffled_indexes: list, mask: np.ndarray, image_len: int) -> None:
        self.number_of_pixels = number_of_pixels
        self.shuffled_indexes = shuffled_indexes
        print(self.shuffled_indexes)
        self.mask = mask
        self.image_len = image_len

    def __call__(self, image: np.ndarray):
        image_copy = copy.deepcopy(image)
        for index in range(self.number_of_pixels):
            # print(len(self.shuffled_indexes))
            i = self.shuffled_indexes[index] // self.image_len
            j = self.shuffled_indexes[index] % self.image_len
            image_copy[i, j, :] = np.add(image_copy[i, j, :],  self.mask[i, j, :])
        return {"image": image_copy.astype(np.uint8)}
