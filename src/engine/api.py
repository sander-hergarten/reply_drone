from dataclasses import dataclass
from torchvision.tv_tensors import Image

@dataclass
class Engine:
    node_positions: list[tuple[float, float, float]]
    node_rotations: list[tuple[float, float, float]]

@dataclass
class SingleNodeState:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    depth_map: Image
    overlap_map: dict[int, float]
    position_of_other_nodes: dict[int, tuple[float, float, float]]
    rotation_of_other_nodes: dict[int, tuple[float, float, float]]
    id: int
