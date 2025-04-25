from dataclasses import dataclass
from torchvision.tv_tensors import Image

"""This dataclass contains the information the RL agent recieves from the environment"""


@dataclass
class SingleNodeState:
    postion: tuple[float, float, float]
    rotation: tuple[float, float, float]
    depth_map: Image
    postion_of_other_nodes: list[tuple[float, float, float]]
    rotation_of_other_nodes: list[tuple[float, float, float]]


@dataclass
class FullGraphState:
    nodes: list[SingleNodeState]
