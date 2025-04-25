from dataclasses import dataclass


@dataclass
class Node:
    id: int
    pos: tuple[float, float, float]
    rot: tuple[float, float, float]


@dataclass
class Collision:
    node_a: Node
    node_b: Node
    collision_points: list[tuple[float, float, float]]
    collision_normals: list[tuple[float, float, float]]


@dataclass
class AllCollisions:
    collisions: list[Collision]
