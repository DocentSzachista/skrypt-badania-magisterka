import numpy as np
import torch

def create_and_shuffle_indexes(matrix_shape: tuple):
    np.random.seed(0)
    indexes = [
        i * 32 + j for i in range(matrix_shape[0]) for j in range(matrix_shape[1])
    ]
    np.random.shuffle(indexes)
    return indexes


def generate_mask(shape: tuple):
    torch.manual_seed(0)
    temp = torch.randn(shape)*255
    return temp.long()


class BaseAugumentation:
    def __init__(self, config: dict) -> None:
        self.name = config.get("name")
        self.start_point = config.get("start_point")
        self.finish_point = config.get("finish_point")
        self.step = config.get("step")

    def make_iterator(self):
        return np.arange(self.start_point, self.finish_point, self.step)


class NoiseAugumentation(BaseAugumentation):
    def __init__(self, config: dict, image_dim: list) -> None:
        super().__init__(config)
        self.mask = generate_mask(list(reversed(image_dim)))
        self.shuffled_indexes = create_and_shuffle_indexes(image_dim)
        self.max_size = image_dim[1] * image_dim[2]
        
class MixupAugumentation(BaseAugumentation):
    def __init__(self, config: dict, image_dim: list) -> None:
        super().__init__(config)
        self.max_size = 100
        self.class_ = config["picked_class"]

