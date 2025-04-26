from scipy.spatial.transform import rotation as R
import torch
import numpy as np
from .model_attributes import MAX_OTHER_NODES, OTHER_NODE_FEATURE_DIM
from typing import Dict, List, Tuple


# Placeholder for layer_init function (assuming it exists as before)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def padding_other_nodes(positions_dict_list, rotations_quat_dict_list, device):
    """
    Helper to process variable-length lists.
    Handles padding and processing (e.g., via Transformer).
    Input: List of position lists, List of rotation lists (one list per batch item)
    Output: Tensor of processed features (Batch, EmbedDim // HEADCOUNT)
    """
    batch_size = len(positions_dict_list)
    # Features will be Quat (4) + Pos (3) = 7
    padded_features = torch.zeros(
        batch_size, MAX_OTHER_NODES, OTHER_NODE_FEATURE_DIM, device=device
    )
    masks = torch.zeros(
        batch_size, MAX_OTHER_NODES, dtype=torch.bool, device=device
    )  # False=valid

    for i in range(batch_size):
        # Extract values (ignore keys/IDs for now)
        # Ensure corresponding pos and rot values are aligned if dict keys differ (shouldn't happen if structure is consistent)
        pos_values = list(positions_dict_list[i].values())
        quat_values = list(
            rotations_quat_dict_list[i].values()
        )  # Already converted to quat tensors

        num_nodes = min(len(pos_values), MAX_OTHER_NODES)

        if num_nodes > 0 and len(pos_values) == len(quat_values):  # Basic check
            pos = torch.tensor(
                pos_values[:num_nodes], dtype=torch.float32, device=device
            )
            # Stack the list of quaternion tensors
            quat = torch.stack(quat_values[:num_nodes]).to(device)

            features = torch.cat([pos, quat], dim=-1)  # Shape: (num_nodes, 7)
            padded_features[i, :num_nodes, :] = features
            masks[i, :num_nodes] = False  # Valid tokens
            # Set remaining masks for padding
            if num_nodes < MAX_OTHER_NODES:
                masks[i, num_nodes:] = True  # Padding tokens

    return padded_features, masks


def filter_by_n_closest(
    origin: tuple[float, float, float],
    positions: dict[int, tuple[float, float, float]],
    rotations: dict[int, tuple[float, float, float]],
    overlap_map: dict[int, float],
    n: int,
):
    smallest_position_ids = list(
        map(
            lambda x: x[0],
            sorted(
                [
                    (i, np.linalg.norm(np.array(origin) - np.array(position)))
                    for i, position in positions.items()
                ],
                key=lambda x: x[1],
            ),
        )
    )[:MAX_OTHER_NODES]

    smallest_positions = {i: positions[i] for i in smallest_position_ids}
    smallest_rotations = {i: rotations[i] for i in smallest_position_ids}
    smallest_overlap = {i: overlap_map[i] for i in smallest_position_ids}

    return smallest_positions, smallest_rotations, smallest_overlap


def curry_euler_to_quaternion_device(device):
    def euler_to_quaternion(rotation: tuple[float, float, float], euler_sequence="xyz"):
        quaternion_scipy = R.from_euler(
            euler_sequence, rotation, degrees=True
        ).as_quat()

        return quaternion_scipy

    return euler_to_quaternion()


def _batch_convert_eul_to_quat(
    eul_rotations: List[Tuple[float, float, float]],  # List of own rotations
    eul_rot_dicts: List[
        Dict[int, Tuple[float, float, float]]
    ],  # List of other node dicts
    euler_seq: str,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[int, torch.Tensor]]]:
    """
    Converts batches of euler rotations (own tuples + dicts of tuples)
    to quaternion tensors using vectorized scipy operations.
    """
    batch_size = len(eul_rotations)

    # Process own rotations
    own_quat_scipy = R.from_euler(euler_seq, eul_rotations, degrees=False).as_quat()
    own_quat_tensor = torch.tensor(
        np.roll(own_quat_scipy, 1, axis=1), dtype=torch.float32, device=device
    )  # [B, 4] (w,x,y,z)

    # Process other node rotations (trickier due to dict structure)
    other_quat_dict_list = []
    for i in range(batch_size):
        eul_dict = eul_rot_dicts[i]
        quat_dict = {}
        if eul_dict:
            ids = list(eul_dict.keys())
            euler_vals = list(eul_dict.values())
            quats_scipy = R.from_euler(euler_seq, euler_vals, degrees=False).as_quat()
            quats_torch = torch.tensor(
                np.roll(quats_scipy, 1, axis=1), dtype=torch.float32, device=device
            )  # [N_nodes, 4]
            for idx, node_id in enumerate(ids):
                quat_dict[node_id] = quats_torch[idx]
        other_quat_dict_list.append(quat_dict)

    return own_quat_tensor, other_quat_dict_list


# Use the original batch-capable padding function structure
def _batch_pad_other_nodes(
    positions_dict_list: List[Dict[int, Tuple[float, float, float]]],
    rotations_quat_dict_list: List[
        Dict[int, torch.Tensor]
    ],  # List of dicts containing QUAT tensors
    max_nodes: int,
    feature_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes lists of filtered dictionaries (batch) into padded tensors and a mask.
    Input: List of filtered pos dicts, List of filtered rot dicts (with QUAT tensors)
    Output: Padded features tensor [B, max_nodes, feature_dim], Mask tensor [B, max_nodes]
    """
    batch_size = len(positions_dict_list)
    # Initialize batch tensors
    padded_features = torch.zeros(batch_size, max_nodes, feature_dim, device=device)
    # Mask: True for padding, False for valid data
    masks = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)

    for i in range(batch_size):
        # Extract values from dicts for this sample
        pos_values = list(positions_dict_list[i].values())
        quat_values = list(rotations_quat_dict_list[i].values())  # List of quat tensors

        num_nodes = min(len(pos_values), max_nodes)

        if num_nodes > 0 and len(pos_values) == len(quat_values):
            pos_tensor = torch.tensor(
                pos_values[:num_nodes], dtype=torch.float32, device=device
            )
            # Stack the list of quaternion tensors
            quat_tensor = torch.stack(quat_values[:num_nodes]).to(device)

            features = torch.cat(
                [pos_tensor, quat_tensor], dim=-1
            )  # Shape: [num_nodes, feature_dim]
            padded_features[i, :num_nodes, :] = features
            masks[i, :num_nodes] = False  # Mark valid data as False
        # If num_nodes < max_nodes, the rest of masks[i] remains True (padding)

    return padded_features, masks


# def _prepare_other_nodes_tensors(
#     position_dict: dict[int, tuple[float, float, float]],
#     rotation_quat_dict: dict[int, torch.Tensor],  # Expects dict with QUATERNION TENSORS
#     max_nodes: int,
#     feature_dim: int,
#     device: torch.device,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Processes filtered dictionaries for a SINGLE state into padded tensors and a mask.
#     Input: Filtered position dict, Filtered rotation dict (with QUATERNION TENSORS)
#     Output: Padded features tensor [max_nodes, feature_dim], Mask tensor [max_nodes]
#     """
#     # Extract values (order based on dict iteration, consistent if Python >= 3.7)
#     pos_values = list(position_dict.values())
#     quat_values = list(rotation_quat_dict.values())  # List of quaternion tensors
#
#     num_nodes = min(len(pos_values), max_nodes)
#
#     # Initialize tensors for a single sample (no batch dimension needed here)
#     padded_features = torch.zeros(max_nodes, feature_dim, device=device)
#     # Mask: True for padding, False for valid data
#     mask = torch.ones(max_nodes, dtype=torch.bool, device=device)
#
#     if num_nodes > 0 and len(pos_values) == len(quat_values):
#         pos_tensor = torch.tensor(
#             pos_values[:num_nodes], dtype=torch.float32, device=device
#         )
#         # Stack the list of quaternion tensors
#         quat_tensor = torch.stack(quat_values[:num_nodes]).to(device)
#
#         features = torch.cat(
#             [pos_tensor, quat_tensor], dim=-1
#         )  # Shape: [num_nodes, feature_dim]
#         padded_features[:num_nodes, :] = features
#         mask[:num_nodes] = False  # Mark valid data as False
#
#     return padded_features, mask
#
#
# def _convert_eul_dict_to_quat_dict(eul_rot_dict, euler_seq, device):
#     """Converts dict values from Euler tuples to Quaternion tensors."""
#     quat_dict = {}
#     if eul_rot_dict:
#         ids = list(eul_rot_dict.keys())
#         euler_vals = list(eul_rot_dict.values())
#         # Convert all Euler rotations at once
#         quats_scipy = R.from_euler(
#             euler_seq, euler_vals, degrees=False
#         ).as_quat()  # Scipy: (x,y,z,w)
#         # Convert to PyTorch Tensor (w,x,y,z) - choose a convention and stick to it
#         quats_torch = torch.tensor(
#             np.roll(quats_scipy, 1, axis=1), dtype=torch.float32, device=device
#         )
#         # Reconstruct dict with original IDs and quaternion tensors
#         for idx, node_id in enumerate(ids):
#             quat_dict[node_id] = quats_torch[idx]
#     return quat_dict
#
#
# def _convert_eul_tuple_to_quat_tensor(eul_tuple, euler_seq, device):
#     """Converts a single Euler tuple to a Quaternion tensor."""
#     quat_scipy = R.from_euler(
#         euler_seq, [eul_tuple], degrees=False
#     ).as_quat()  # Scipy: (x,y,z,w)
#     quat_torch = torch.tensor(
#         np.roll(quat_scipy, 1, axis=1)[0], dtype=torch.float32, device=device
#     )  # (w,x,y,z)
#     return quat_torch
