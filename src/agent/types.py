from dataclasses import dataclass
from torchvision.tv_tensors import Image

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


@dataclass
class FullGraphState:
    id_to_node_map: dict[int, SingleNodeState]
