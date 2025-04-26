# agent/types.py

from dataclasses import dataclass
from torchvision.tv_tensors import Image  # Assuming depth map is an Image tensor

"""
This module defines data structures for representing agent states and actions,
primarily for communication between the environment and the trainer/agent.
"""


@dataclass
class SingleNodeState:
    """
    Represents the state information for a single agent/node received
    from the environment at a particular timestep.
    """

    id: int  # Unique identifier for this agent/node

    position: tuple[float, float, float]  # (x, y, z) coordinates
    rotation: tuple[float, float, float]
    depth_map: Image  # Depth map observation (e.g., a torch tensor)
    position_of_other_nodes: dict[int, tuple[float, float, float]]
    rotation_of_other_nodes: dict[int, tuple[float, float, float]]


# Type alias for the full state of the environment, containing states for all nodes.
FullGraphState = dict[int, SingleNodeState]


@dataclass
class SingleAction:
    """
    Represents the action command for a single agent/node to be sent
    to the environment.
    """

    id: int  # Identifier for the agent/node this action applies to

    # Desired new state components
    new_pos: tuple[float, float, float]  # Target (x, y, z) position
    # Target rotation sent to the environment.
    # The trainer constructs this as an Euler tuple (e.g., (0, 0, target_z_angle))
    # based on the agent's single Z-angle output and the expected 'euler_seq'.
    new_rot: tuple[float, float, float]

    # 'done' field was present but seemed unused in the provided trainer logic.
    # Keep commented out unless it's required by the specific environment implementation.
    # done: bool
