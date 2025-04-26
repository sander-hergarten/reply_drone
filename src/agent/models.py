import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_attributes import EMBEDDING_DIM, HEADCOUNT, QUATERNION_DIM
from .utils import (
    curry_euler_to_quaternion_device,
    filter_by_n_closest,
    layer_init,
    padding_other_nodes,
)
from .processors import vector_processor, cnn_processor, other_nodes_processor
from .types import SingleNodeState


class Agent(nn.Module):
    def __init__(self, action_space_shape, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        action_dim = np.prod(action_space_shape)

        # --- 1. Input Processing Modules ---
        # Module for own position and rotation (simple vectors)
        # Module for depth map (CNN) - Define your own CNN architecture
        # Module for variable-length lists of other nodes
        self.vector_processor = vector_processor
        self.cnn_processor = cnn_processor
        self.other_nodes_processor = other_nodes_processor

        # Option A: Padding + Transformer Encoder (Powerful, handles relations)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=4,
            dim_feedforward=EMBEDDING_DIM * 2,
            batch_first=True,  # Use batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # After Transformer, aggregate sequence: Use mean pooling over sequence length dim
        # Output will be (Batch, EmbedDim)
        self.transformer_output_processor = nn.Sequential(
            layer_init(
                nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // HEADCOUNT)
            )  # Output partial embedding
        )

        # --- 2. Feature Combination ---
        combined_embedding_dim = (EMBEDDING_DIM // HEADCOUNT) * 3

        self.combiner = nn.Sequential(
            layer_init(nn.Linear(combined_embedding_dim, 256)), nn.ReLU()
        )

        # --- 3. Output Heads ---
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def _process_other_nodes(self, padded_features, masks, device):
        """
        Helper to process variable-length lists.
        Handles processing (e.g., via Transformer).
        Input: List of position lists, List of rotation lists (one list per batch item)
        Output: Tensor of processed features (Batch, EmbedDim // 3)
        """

        # ... (Rest of transformer processing is the same) ...
        x = self.other_nodes_processor(padded_features)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=masks)
        transformer_output = transformer_output.masked_fill(masks.unsqueeze(-1), 0.0)
        non_pad_count = (~masks).sum(dim=1, keepdim=True)
        summed_features = transformer_output.sum(dim=1)
        mean_features = torch.where(
            non_pad_count > 0,
            summed_features / non_pad_count,
            torch.zeros_like(summed_features),
        )
        processed_other_nodes = self.transformer_output_processor(mean_features)
        return processed_other_nodes

    def get_value(self, state: SingleNodeState):
        # This function now needs to process the structured state
        # It's easier to just call get_action_and_value and extract the value
        _, _, _, value = self.get_action_and_value(state)
        return value

    def get_action_and_value(self, state_batch: list[SingleNodeState], action=None):
        device = self.actor_logstd.device
        euler_to_quaternion = curry_euler_to_quaternion_device(device)

        # Own rotation conversion
        own_quaternions = map(euler_to_quaternion, [s.rotation for s in state_batch])
        own_quaternions = torch.tensor(
            np.roll(own_quaternions, 1, axis=1),
            dtype=torch.float32,
            device=device,
        )  # (w,x,y,z)

        batched_filtered_position = []
        batched_filtered_rotation = []
        batched_filtered_overlap = []

        for batch in state_batch:
            filtered_position, filtered_rotation, filtered_overlap = (
                filter_by_n_closest(
                    batch.position,
                    batch.position_of_other_nodes,
                    batch.rotation_of_other_nodes,
                    batch.overlap_map,
                )
            )

            ids = filtered_rotation.keys()
            filtered_quaternion_scipy = list(
                map(euler_to_quaternion, filtered_rotation.values())
            )
            filtered_quaternion_torch = torch.tensor(
                np.roll(filtered_quaternion_scipy, 1, axis=1), dtype=torch.float32
            )  # (w,x,y,z)

            filtered_quaternion_dict = {
                node_id: filtered_quaternion_torch[idx]
                for idx, node_id in enumerate(ids)
            }
            batched_filtered_position.append(filtered_position)
            batched_filtered_rotation.append(filtered_quaternion_dict)
            batched_filtered_overlap.append(filtered_overlap)

        # Extract other components (same as before)
        pos_batch = torch.tensor(
            [s.position for s in state_batch], dtype=torch.float32, device=device
        )
        depth_batch = torch.stack([s.depth_map for s in state_batch]).to(device)

        # 1. Process inputs (Now using quaternions internally)
        vector_input = torch.cat(
            [pos_batch, own_quaternions], dim=-1
        )  # Use converted quat

        vector_features = self.vector_processor(vector_input)
        cnn_features = self.cnn_processor(depth_batch)
        # Pass list of position dicts and list of quaternion dicts to helper
        other_nodes_features = self._process_other_nodes(
            *padding_other_nodes(
                batched_filtered_position, batched_filtered_rotation, device
            )
        )

        # --- Steps 2 & 3 (Combine features, Get outputs) are the same as before ---
        # ... (Combine features) ...
        # ... (Get action_mean_raw, value) ...
        # ... (Normalize quaternion part of action_mean) ...
        # ... (Sample action, normalize quaternion part) ...
        # ... (Calculate log_prob, entropy) ...

        # Ensure the return signature is correct
        combined_features = torch.cat(
            [vector_features, cnn_features, other_nodes_features], dim=-1
        )
        combined_embedding = self.combiner(combined_features)

        action_mean_raw = self.actor_mean(combined_embedding)
        value = self.critic(combined_embedding)  # (B, 1)

        action_mean_pos = action_mean_raw[..., :-QUATERNION_DIM]
        action_mean_quat_raw = action_mean_raw[..., -QUATERNION_DIM:]
        action_mean_quat = F.normalize(action_mean_quat_raw, p=2, dim=-1)
        action_mean = torch.cat([action_mean_pos, action_mean_quat], dim=-1)

        action_logstd_expanded = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd_expanded)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action_sampled_raw = probs.sample()
            action_pos = action_sampled_raw[..., :-QUATERNION_DIM]
            action_quat_raw = action_sampled_raw[..., -QUATERNION_DIM:]
            action_quat_normalized = F.normalize(action_quat_raw, p=2, dim=-1)
            action = torch.cat([action_pos, action_quat_normalized], dim=-1)

        probs_for_logprob = torch.distributions.Normal(action_mean_raw, action_std)
        log_prob = probs_for_logprob.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        return action, log_prob, entropy, value.flatten()


# Example usage (conceptual, assuming env and batching are handled)
# action_dim = ... # Define based on your environment's action space
# agent = Agent(action_space_shape=(action_dim,), rpo_alpha=args.rpo_alpha).to(device)

# During training loop (after collecting data and creating a minibatch 'mb_state_batch')
# mb_state_batch = ... # Your custom batch object/dict/list for the minibatch
# actions_in_batch = ... # Corresponding actions from the batch
# _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_state_batch, actions_in_batch)
