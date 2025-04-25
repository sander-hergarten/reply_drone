from dataclasses import dataclass
from torchvision.tv_tensors import Image

"""This dataclass contains the information the barcode classifier recieves from the environment"""


@dataclass
class SingleBarcodeClassifier:
    image_dim: tuple[int, int]
    image: Image
    id: int


@dataclass
class AllBarcodeClassifiers:
    id_to_barcode_map: dict[int, SingleBarcodeClassifier]
