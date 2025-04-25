from dataclasses import dataclass


@dataclass
class SingleAction:
    id: int
    new_pos: tuple[float, float, float]
    new_rot: tuple[float, float, float]
    done: bool


@dataclass
class AllActions:
    id_to_action_map: dict[int, SingleAction]
