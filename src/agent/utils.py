# agent/utils.py

from scipy.spatial.transform import (
    rotation as R,
)  # Still needed for Euler processing if env uses it
import torch
import numpy as np
from .model_attributes import (
    MAX_OTHER_NODES,
    OTHER_NODE_FEATURE_DIM,  # Should reflect Pos + RotAngle
    POSITION_DIM,
    ROTATION_DIM,  # Single Z-angle dimension
)
from typing import Dict, List, Tuple


# Initialization function for neural network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights of a layer using orthogonal initialization
    and sets biases to a constant value.

    Args:
        layer: The neural network layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.

    Returns:
        The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# --- MODIFIED: Function to extract Z-rotation from Euler ---
def _batch_extract_z_rotation(
    eul_rotations: List[
        Tuple[float, float, float]
    ],  # List of own Euler rotations from env state
    eul_rot_dicts: List[
        Dict[int, Tuple[float, float, float]]
    ],  # List of other node Euler dicts from env state
    euler_seq_indices: Tuple[
        int, int, int
    ],  # Indices corresponding to axes (e.g., (0,1,2) for 'xyz')
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[int, torch.Tensor]]]:
    """
    Extracts the Z-axis rotation angle from batches of Euler rotations.
    Assumes the environment provides Euler angles and the desired Z-axis
    index is known from euler_seq_indices.

    Args:
        eul_rotations: List of [BatchSize] tuples, each containing (eul_x, eul_y, eul_z) or similar.
        eul_rot_dicts: List of [BatchSize] dictionaries, mapping node_id to Euler tuple.
        euler_seq_indices: A tuple indicating the order of axes (e.g., (0, 1, 2) for 'xyz').
                           The third element indicates the index of the Z rotation.
        device: The torch device to place the output tensors on.

    Returns:
        Tuple containing:
        - own_z_rot_tensor: Tensor of shape [BatchSize, 1] containing Z-rotation angles for own agent.
        - other_z_rot_dict_list: List of [BatchSize] dictionaries, mapping node_id to Z-rotation tensor [1].
    """
    batch_size = len(eul_rotations)
    # Determine the index for the Z-axis based on the provided Euler sequence order
    z_index = euler_seq_indices[2]  # e.g., if 'xyz', indices=(0,1,2), z_index=2

    # Process own rotations
    own_rot_np = np.array(eul_rotations, dtype=np.float32)
    # Extract the Z-rotation angle using the determined index
    own_z_rot_tensor = torch.tensor(
        own_rot_np[:, z_index : z_index + 1], device=device
    )  # Shape [B, 1]

    # Process other node rotations
    other_z_rot_dict_list = []
    for i in range(batch_size):
        eul_dict = eul_rot_dicts[i]
        z_rot_dict = {}
        if eul_dict:  # Check if the dictionary is not empty
            ids = list(eul_dict.keys())
            # Convert list of Euler tuples to a NumPy array
            euler_vals = np.array(list(eul_dict.values()), dtype=np.float32)
            # Extract the Z-rotation angles for all nodes in this batch item
            z_rots_torch = torch.tensor(
                euler_vals[:, z_index : z_index + 1], device=device
            )  # Shape [N_nodes, 1]
            # Reconstruct the dictionary with node IDs and their corresponding Z-rotation tensors
            for idx, node_id in enumerate(ids):
                z_rot_dict[node_id] = z_rots_torch[
                    idx
                ]  # Each value is a tensor of shape [1]
        other_z_rot_dict_list.append(z_rot_dict)

    return own_z_rot_tensor, other_z_rot_dict_list


# --- END MODIFIED ---


# --- MODIFIED: Adapt padding function for Z-angle ---
def _batch_pad_other_nodes(
    positions_dict_list: List[Dict[int, Tuple[float, float, float]]],
    rotations_z_angle_dict_list: List[
        Dict[int, torch.Tensor]
    ],  # Expects dicts with Z-ANGLE tensors [1]
    max_nodes: int,
    feature_dim: int,  # Should be POSITION_DIM + ROTATION_DIM
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes lists of filtered position and Z-angle rotation dictionaries (batch)
    into padded feature tensors and a corresponding mask tensor for use with Transformer.

    Args:
        positions_dict_list: List [BatchSize] of dictionaries mapping node_id to position tuple.
        rotations_z_angle_dict_list: List [BatchSize] of dictionaries mapping node_id to Z-angle tensor [1].
        max_nodes: The maximum number of nodes to include (sequence length for Transformer).
        feature_dim: The expected feature dimension per node (POSITION_DIM + ROTATION_DIM).
        device: The torch device for output tensors.

    Returns:
        Tuple containing:
        - padded_features: Tensor of shape [BatchSize, max_nodes, feature_dim] with zero-padding.
        - masks: Boolean tensor of shape [BatchSize, max_nodes] where True indicates padding.
    """
    batch_size = len(positions_dict_list)
    # Initialize tensors for the batch with zeros for features and ones for mask (True=padding)
    padded_features = torch.zeros(batch_size, max_nodes, feature_dim, device=device)
    masks = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)

    for i in range(batch_size):
        # Extract position tuples and Z-angle tensors for the current batch item
        pos_values = list(positions_dict_list[i].values())
        z_angle_values = list(
            rotations_z_angle_dict_list[i].values()
        )  # List of tensors [1]

        # Determine the number of nodes to process (up to max_nodes)
        num_nodes = min(len(pos_values), max_nodes)

        # Ensure consistency and proceed if there are nodes to process
        if num_nodes > 0 and len(pos_values) == len(z_angle_values):
            # Convert position tuples to a tensor
            pos_tensor = torch.tensor(
                pos_values[:num_nodes], dtype=torch.float32, device=device
            )  # Shape [num_nodes, POSITION_DIM]

            # Stack the list of Z-angle tensors (each shape [1]) into a single tensor
            z_angle_tensor = torch.stack(z_angle_values[:num_nodes]).to(
                device
            )  # Shape [num_nodes, ROTATION_DIM=1]

            # Concatenate position and Z-angle tensors to form the feature vector for each node
            features = torch.cat(
                [pos_tensor, z_angle_tensor], dim=-1
            )  # Shape: [num_nodes, feature_dim]

            # Fill the padded_features tensor with the actual features
            padded_features[i, :num_nodes, :] = features
            # Update the mask: False indicates valid data, True indicates padding
            masks[i, :num_nodes] = False
        # If num_nodes < max_nodes, the remaining elements of masks[i] stay True (padding)

    return padded_features, masks


# --- END MODIFIED ---


def filter_by_n_closest(
    origin: tuple[float, float, float],
    positions: dict[int, tuple[float, float, float]],
    rotations: dict[
        int, tuple[float, float, float]
    ],  # Assume this contains Euler angles from env
    n: int,  # Use the passed argument 'n' instead of hardcoding MAX_OTHER_NODES
) -> Tuple[
    Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]]
]:
    """
    Filters the 'positions' and 'rotations' dictionaries to keep only the 'n'
    nodes closest to the 'origin' position.

    Args:
        origin: The (x, y, z) position of the reference agent.
        positions: Dictionary mapping node_id to (x, y, z) position tuple.
        rotations: Dictionary mapping node_id to (e.g., Euler) rotation tuple.
                   The keys must correspond to the keys in 'positions'.
        n: The maximum number of closest nodes to keep.

    Returns:
        Tuple containing:
        - smallest_positions: Filtered dictionary of positions.
        - smallest_rotations: Filtered dictionary of rotations corresponding to the closest positions.
    """
    # Calculate Euclidean distance from origin to each node position
    distances = [
        (node_id, np.linalg.norm(np.array(origin) - np.array(pos)))
        for node_id, pos in positions.items()
    ]

    # Sort nodes by distance (ascending)
    sorted_distances = sorted(distances, key=lambda x: x[1])

    # Get the IDs of the 'n' closest nodes
    smallest_position_ids = [item[0] for item in sorted_distances[:n]]  # Use n here

    # Create new dictionaries containing only the data for the closest nodes
    smallest_positions = {i: positions[i] for i in smallest_position_ids}
    smallest_rotations = {i: rotations[i] for i in smallest_position_ids}

    return smallest_positions, smallest_rotations


# --- Removed functions related to quaternion conversion as they are no longer needed ---
# def curry_euler_to_quaternion_device(device): ...
# def _batch_convert_eul_to_quat(...): ... # Replaced by _batch_extract_z_rotation
# def _prepare_other_nodes_tensors(...): ... # Logic integrated into _batch_pad_other_nodes
# def _convert_eul_dict_to_quat_dict(...): ...
# def _convert_eul_tuple_to_quat_tensor(...): ...
