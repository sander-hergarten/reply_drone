from dataclasses import dataclass
from torchvision.tv_tensors import Image
from typing import Callable


@dataclass
class SingleBarcodeClassifier:
    image_dim: tuple[int, int]
    image: Image
    id: int


AllBarcodeClassifiers = dict[int, SingleBarcodeClassifier]


#########################
# TO IMPLEMENT:
get_confidence_scores = Callable[[AllBarcodeClassifiers], dict[int, float]]
#########################
