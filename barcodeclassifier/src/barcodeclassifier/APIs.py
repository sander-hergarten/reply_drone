from dataclasses import dataclass

"""This dataclass contains the information the RL agent recieves from the barcode classifier."""


@dataclass
class SingleBarcodeClassifierConfidence:
    confidence_score: float
    id: int


@dataclass
class AllBarcodeClassifierConfidences:
    id_to_confidence_map: dict[int, SingleBarcodeClassifierConfidence]

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
