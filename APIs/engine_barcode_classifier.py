from dataclasses import dataclass
from torchvision.tv_tensors import Image

"""This dataclass contains the information the barcode classifier recieves from the environment"""


@dataclass
class BarcodeClassifier:
    image_dim: tuple[int, int]
    image: Image
