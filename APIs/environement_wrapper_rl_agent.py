from dataclasses import dataclass
from torchvision.tv_tensors import Image
from typing import Callable

"""This dataclass contains the information the RL agent recieves from the environment"""


@dataclass
class SingleNodeStateNoOverlap:
    postion: tuple[float, float, float]
    rotation: tuple[float, float, float]
    depth_map: Image
    postion_of_other_nodes: dict[int, tuple[float, float, float]]
    rotation_of_other_nodes: dict[int, tuple[float, float, float]]
    id: int


FullGraphStateNoOverlap = dict[int, SingleNodeStateNoOverlap]


AllRewardScores = dict[int, float]


@dataclass
class SingleAction:
    id: int
    new_pos: tuple[float, float, float]
    new_rot: tuple[float, float, float]
    done: bool


AllActions = dict[int, SingleAction]


#########################
# TO IMPLEMENT:
reset = Callable[[], tuple[FullGraphStateNoOverlap, AllRewardScores]]
step = Callable[
    [AllActions],
    tuple[FullGraphStateNoOverlap, AllRewardScores],
]
#########################
