from dataclasses import dataclass
from torchvision.tv_tensors import Image
from typing import Callable

"""This dataclass contains the information the RL agent recieves from the environment"""


@dataclass
class SingleNodeState:
    postion: tuple[float, float, float]
    rotation: tuple[float, float, float]
    depth_map: Image
    overlap_map: dict[int, float]
    postion_of_other_nodes: dict[int, tuple[float, float, float]]
    rotation_of_other_nodes: dict[int, tuple[float, float, float]]
    id: int


FullGraphState = dict[int, SingleNodeState]


@dataclass
class SingleBarcodeClassifier:
    image_dim: tuple[int, int]
    image: Image
    id: int


AllBarcodeClassifiers = dict[int, SingleBarcodeClassifier]


@dataclass
class Engine:
    node_positions: list[tuple[float, float, float]]
    node_rotations: list[tuple[float, float, float]]


@dataclass
class SingleAction:
    id: int
    delta_pos: tuple[float, float, float]
    delta_rot: tuple[float, float, float]
    done: bool


AllActions = dict[int, SingleAction]


#########################
# TO IMPLEMENT:
reset = Callable[[int], tuple[FullGraphState, AllBarcodeClassifiers]]
step = Callable[
    [AllActions],
    tuple[FullGraphState, AllBarcodeClassifiers],
]
#########################
