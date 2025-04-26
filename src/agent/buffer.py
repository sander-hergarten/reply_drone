import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer  # For type hinting
from typing import List

from .types import SingleNodeState
from .utils import (
    filter_by_n_closest,
    _batch_convert_eul_to_quat,
    _batch_pad_other_nodes,
)
from .model_attributes import MAX_OTHER_NODES, OTHER_NODE_FEATURE_DIM


def add_batch_transitions_to_buffer(
    replay_buffer: ReplayBuffer,
    current_states: List[SingleNodeState],
    actions_agent: torch.Tensor,  # Batch: [B, action_dim] (contains quaternion)
    rewards: torch.Tensor,  # Batch: [B] or [B, 1]
    dones: torch.Tensor,  # Batch: [B] or [B, 1] (bool or float)
    next_states: List[SingleNodeState],
    max_other_nodes: int = MAX_OTHER_NODES,  # Use defaults or pass args
    other_node_feature_dim: int = OTHER_NODE_FEATURE_DIM,
    device: torch.device = torch.device("cpu"),  # Default to CPU if not specified
    euler_seq: str = "xyz",
):
    """
    Processes a BATCH of transitions (current_state -> next_state) and adds them
    to the replay buffer using the padding approach for variable data.

    Args:
        replay_buffer: The torchrl.data.ReplayBuffer instance.
        current_states: List of SingleNodeState objects for the current timesteps.
        actions_agent: Batch of action tensors from agent (includes quaternion).
        rewards: Batch of scalar rewards received.
        dones: Batch of boolean done flags for the current timesteps.
        next_states: List of SingleNodeState objects for the next timesteps.
        max_other_nodes: Max neighbours for filtering and padding.
        other_node_feature_dim: Dimension for padded features (pos+quat).
        device: Torch device.
        euler_seq: Euler sequence string (e.g., 'xyz') for conversions.
    """
    batch_size = len(current_states)
    if batch_size == 0:
        return  # Nothing to add

    # Ensure rewards and dones have the right shape (B, 1) for TensorDict
    rewards = rewards.reshape(-1, 1).to(dtype=torch.float32, device=device)
    # Ensure dones are boolean and shaped [B, 1]
    dones = dones.reshape(-1, 1).to(dtype=torch.bool, device=device)
    # Move actions to device
    actions_agent = actions_agent.detach().to(device)

    with torch.no_grad():  # Ensure no gradients are computed during data prep
        # --- Process Batch of CURRENT States ---
        # 1. Filter other nodes (per sample)
        filtered_pos_curr_list = []
        filtered_rot_eul_curr_list = []
        for i in range(batch_size):
            f_pos, f_rot, _ = filter_by_n_closest(
                current_states[i].position,
                current_states[i].position_of_other_nodes,
                current_states[i].rotation_of_other_nodes,
                current_states[i].overlap_map,
                max_nodes=max_other_nodes,
            )
            filtered_pos_curr_list.append(f_pos)
            filtered_rot_eul_curr_list.append(f_rot)

        # 2. Convert all rotations to Quaternions (vectorized where possible)
        own_rot_eul_curr_list = [s.rotation for s in current_states]
        own_rot_quat_curr_batch, other_rot_quat_dict_curr_list = (
            _batch_convert_eul_to_quat(
                own_rot_eul_curr_list, filtered_rot_eul_curr_list, euler_seq, device
            )
        )

        # 3. Pad other nodes data (batch)
        other_nodes_padded_curr_batch, other_nodes_mask_curr_batch = (
            _batch_pad_other_nodes(
                filtered_pos_curr_list,  # List of filtered pos dicts
                other_rot_quat_dict_curr_list,  # List of filtered quat dicts
                max_nodes=max_other_nodes,
                feature_dim=other_node_feature_dim,
                device=device,
            )
        )

        # 4. Batch fixed-size components
        pos_curr_batch = torch.tensor(
            [s.position for s in current_states], dtype=torch.float32, device=device
        )
        depth_curr_batch = torch.stack([s.depth_map for s in current_states]).to(
            device
        )  # Assuming depth maps are stackable tensors

        # --- Process Batch of NEXT States (Similar steps) ---
        # 1. Filter
        filtered_pos_next_list = []
        filtered_rot_eul_next_list = []
        for i in range(batch_size):
            f_pos, f_rot, _ = filter_by_n_closest(
                next_states[i].position,
                next_states[i].position_of_other_nodes,
                next_states[i].rotation_of_other_nodes,
                next_states[i].overlap_map,
                max_nodes=max_other_nodes,
            )
            filtered_pos_next_list.append(f_pos)
            filtered_rot_eul_next_list.append(f_rot)

        # 2. Convert rotations
        own_rot_eul_next_list = [s.rotation for s in next_states]
        own_rot_quat_next_batch, other_rot_quat_dict_next_list = (
            _batch_convert_eul_to_quat(
                own_rot_eul_next_list, filtered_rot_eul_next_list, euler_seq, device
            )
        )

        # 3. Pad other nodes data
        other_nodes_padded_next_batch, other_nodes_mask_next_batch = (
            _batch_pad_other_nodes(
                filtered_pos_next_list,
                other_rot_quat_dict_next_list,
                max_nodes=max_other_nodes,
                feature_dim=other_node_feature_dim,
                device=device,
            )
        )

        # 4. Batch fixed-size components
        pos_next_batch = torch.tensor(
            [s.position for s in next_states], dtype=torch.float32, device=device
        )
        depth_next_batch = torch.stack([s.depth_map for s in next_states]).to(device)

        # --- Create BATCH TensorDict for the transition ---
        batch_td = TensorDict(
            {
                "observation": TensorDict(
                    {
                        "position": pos_curr_batch,  # [B, 3]
                        "rotation_quat": own_rot_quat_curr_batch,  # [B, 4]
                        "depth_map": depth_curr_batch,  # [B, C, H, W]
                        "other_nodes_features": other_nodes_padded_curr_batch,  # [B, max_nodes, feat_dim]
                        "other_nodes_mask": other_nodes_mask_curr_batch,  # [B, max_nodes]
                    },
                    batch_size=[batch_size],
                ),
                "action": actions_agent,  # [B, action_dim]
                # Use "next" key for next state observation, reward, done
                "next": TensorDict(
                    {
                        "observation": TensorDict(
                            {
                                "position": pos_next_batch,
                                "rotation_quat": own_rot_quat_next_batch,
                                "depth_map": depth_next_batch,
                                "other_nodes_features": other_nodes_padded_next_batch,
                                "other_nodes_mask": other_nodes_mask_next_batch,
                            },
                            batch_size=[batch_size],
                        ),
                        "reward": rewards,  # [B, 1]
                        "done": dones,  # [B, 1] (bool)
                        # Consider adding "terminated" and "truncated" if your env provides them
                        # "terminated": dones, # Placeholder, use actual terminated flags
                        # "truncated": torch.zeros_like(dones), # Placeholder, use actual truncated flags
                    },
                    batch_size=[batch_size],
                ),
            },
            batch_size=[batch_size],
        )  # Outer TensorDict representing the batch

        # Add the prepared BATCH TensorDict to the replay buffer
        replay_buffer.add(batch_td)
