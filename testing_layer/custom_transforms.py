import numpy as np
import copy




class NoiseTransform(object):

    def __init__(self, number_of_pixels: int, shuffled_indexes: list, mask: np.ndarray) -> None:
        self.number_of_pixels = number_of_pixels
        self.shuffled_indexes = shuffled_indexes
        self.mask = mask


    def __call__(self, image: np.ndarray):
        image_copy = copy.deepcopy(image)
        print("akuku")
        print(image.shape)
        image_length = 65536
        for index in range(self.number_of_pixels):
            i = self.shuffled_indexes[index] // image_length
            j = self.shuffled_indexes[index] % image_length
            image_copy[i,j, :] = np.add(image_copy[i,j, :],  self.mask[i,j, :])
        return {"image": image_copy.astype(np.uint8)}