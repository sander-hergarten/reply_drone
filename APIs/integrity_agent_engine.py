from dataclasses import dataclass


@dataclass
class IntegrityPoints:
    node_positions: list[tuple[float, float, float]]
    node_rotations: list[tuple[float, float, float]]
