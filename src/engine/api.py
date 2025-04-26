from dataclasses import dataclass
from torchvision.tv_tensors import Image
from typing import Callable

@dataclass
class SingleNodeState:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    depth_map: Image
    overlap_map: dict[int, float]
    position_of_other_nodes: dict[int, tuple[float, float, float]]
    rotation_of_other_nodes: dict[int, tuple[float, float, float]]
    id: int

FullGraphState = dict[int, SingleNodeState]

@dataclass
class Engine:
    node_positions: list[tuple[float, float, float]]
    node_rotations: list[tuple[float, float, float]]

@dataclass
class SingleAction:
    id: int
    delta_pos: tuple[float, float, float]
    delta_rot: tuple[float, float, float]

@dataclass
class SingleBarcodeClassifier:
    image_dim: tuple[int, int]
    image: Image
    id: int

AllBarcodeClassifiers = dict[int, SingleBarcodeClassifier]

AllActions = dict[int, SingleAction]

reset = Callable[
    [int],
    tuple[FullGraphState, AllBarcodeClassifiers]
]
step = Callable[
    [AllActions],
    tuple[FullGraphState, AllBarcodeClassifiers],
]

get_confidence_scores = Callable[[AllBarcodeClassifiers], dict[int, float]]
