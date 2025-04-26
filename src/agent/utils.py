from scipy.spatial.transform import rotation as R
import torch
import numpy as np
from .model_attributes import MAX_OTHER_NODES, OTHER_NODE_FEATURE_DIM


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
