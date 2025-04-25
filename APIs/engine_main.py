from dataclasses import dataclass


@dataclass
class Engine:
    node_positions: list[tuple[float, float, float]]
    node_rotations: list[tuple[float, float, float]]
