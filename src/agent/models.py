# agent/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import (
    Dict,
    Optional,
    Tuple,
    List,
)  # Added List for potential type hints elsewhere

# Import model attributes and helper functions
from .model_attributes import (
    EMBEDDING_DIM,
    HEADCOUNT,
    MAX_OTHER_NODES,
    QUATERNION_DIM,
    OTHER_NODE_FEATURE_DIM,
    VECTOR_INPUT_DIM,  # Needed for internal consistency checks if desired
)
from .utils import layer_init  # Removed utils not directly used by Agent class anymore

# Import network components
from .processors import vector_processor, cnn_processor, other_nodes_processor

# Import type definition for clarity, although not used in function signatures anymore
from .types import SingleNodeState


class Agent(nn.Module):
    def __init__(self, action_space_shape, rpo_alpha):
        """
        Initializes the Agent network.

        Args:
            action_space_shape (tuple or list): Shape of the action space (determines output size).
                                                Expected to be the agent's action dimension (e.g., including quaternion).
            rpo_alpha (float): Alpha parameter for RPO noise injection.
        """
        super().__init__()
        self.rpo_alpha = rpo_alpha
        # Calculate action dimension from the provided shape
        # Example: if shape is (7,), action_dim is 7
        action_dim = np.prod(action_space_shape).item()  # Use .item() for scalar

        # --- 1. Input Processing Modules (Imported) ---
        self.vector_processor = vector_processor
        self.cnn_processor = cnn_processor
        # This processes the features extracted from padded tensors
        self.other_nodes_feature_processor = other_nodes_processor

        # --- Transformer for processing sequence of other nodes ---
        # Assuming the other_nodes_processor above prepares input for the Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,  # Input embedding dim from other_nodes_processor
            nhead=4,  # Should be a divisor of EMBEDDING_DIM
            dim_feedforward=EMBEDDING_DIM * 2,
            batch_first=True,  # Crucial: Input shape (Batch, SeqLen, EmbedDim)
            # dropout=0.1 # Optional dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- Aggregation/Output head for Transformer sequence ---
        self.transformer_output_processor = nn.Sequential(
            layer_init(
                nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // HEADCOUNT)
            )  # Output partial embedding
        )

        # --- 2. Feature Combination ---
        # Ensure HEADCOUNT divides EMBEDDING_DIM cleanly
        if EMBEDDING_DIM % HEADCOUNT != 0:
            raise ValueError(
                f"EMBEDDING_DIM ({EMBEDDING_DIM}) must be divisible by HEADCOUNT ({HEADCOUNT})"
            )
        combined_embedding_dim = (
            EMBEDDING_DIM // HEADCOUNT
        ) * 3  # vector, cnn, other_nodes

        self.combiner = nn.Sequential(
            layer_init(nn.Linear(combined_embedding_dim, 256)), nn.ReLU()
        )

        # --- 3. Output Heads ---
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def _process_other_nodes(
        self,
        other_nodes_features: torch.Tensor,  # Input: [B, max_nodes, feature_dim=7]
        other_nodes_mask: torch.Tensor,  # Input: [B, max_nodes] (True=padding)
    ) -> torch.Tensor:
        """
        Processes the padded batch of other node features using Transformer and aggregation.

        Args:
            other_nodes_features: Padded tensor of features (pos+quat) for other nodes.
            other_nodes_mask: Boolean mask indicating padded elements (True for padding).

        Returns:
            torch.Tensor: Processed embedding for other nodes [B, EMBEDDING_DIM // HEADCOUNT].
        """
        # 1. Project features to embedding dimension
        # Input: [B, max_nodes, 7] -> Output: [B, max_nodes, EMBEDDING_DIM]
        projected_features = self.other_nodes_feature_processor(other_nodes_features)

        # 2. Apply Transformer Encoder
        # Input: [B, max_nodes, EMBEDDING_DIM], Mask: [B, max_nodes] (True=pad)
        # Output: [B, max_nodes, EMBEDDING_DIM]
        transformer_output = self.transformer_encoder(
            projected_features, src_key_padding_mask=other_nodes_mask
        )

        # 3. Aggregate Transformer output (masked mean pooling)
        # Zero out embeddings corresponding to padding
        transformer_output = transformer_output.masked_fill(
            other_nodes_mask.unsqueeze(-1), 0.0
        )
        # Count non-padded elements per batch item for averaging
        non_pad_count = (~other_nodes_mask).sum(dim=1, keepdim=True)  # Shape: [B, 1]
        # Sum features across the sequence dimension
        summed_features = transformer_output.sum(dim=1)  # Shape: [B, EMBEDDING_DIM]
        # Calculate mean, avoiding division by zero if all nodes were padded
        mean_features = torch.where(
            non_pad_count > 0,
            summed_features / non_pad_count.clamp(min=1),  # Clamp to avoid NaN
            torch.zeros_like(summed_features),
        )  # Shape: [B, EMBEDDING_DIM]

        # 4. Final processing layer for other nodes embedding
        # Input: [B, EMBEDDING_DIM] -> Output: [B, EMBEDDING_DIM // HEADCOUNT]
        processed_other_nodes = self.transformer_output_processor(mean_features)

        return processed_other_nodes

    def get_value(self, processed_state_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the value estimate for a batch of pre-processed states.

        Args:
            processed_state_batch (Dict[str, torch.Tensor]): Dictionary containing
                batched tensors for "position", "rotation_quat", "depth_map",
                "other_nodes_features", "other_nodes_mask".

        Returns:
            torch.Tensor: Value estimates for the batch [B, 1].
        """
        # Simplified call, just need the forward pass logic up to the critic
        # Extract features from the pre-processed batch
        pos_batch = processed_state_batch["position"]
        rot_quat_batch = processed_state_batch["rotation_quat"]
        depth_batch = processed_state_batch["depth_map"]
        other_nodes_padded_batch = processed_state_batch["other_nodes_features"]
        other_nodes_mask_batch = processed_state_batch["other_nodes_mask"]

        # 1. Process inputs through respective modules
        vector_input = torch.cat([pos_batch, rot_quat_batch], dim=-1)
        vector_features = self.vector_processor(vector_input)
        cnn_features = self.cnn_processor(depth_batch)
        other_nodes_features = self._process_other_nodes(
            other_nodes_padded_batch, other_nodes_mask_batch
        )

        # 2. Combine features
        combined_features = torch.cat(
            [vector_features, cnn_features, other_nodes_features], dim=-1
        )
        combined_embedding = self.combiner(combined_features)

        # 3. Get value estimate
        value = self.critic(combined_embedding)
        return value  # Shape [B, 1]

    def get_action_and_value(
        self,
        processed_state_batch: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,  # Optional action for RPO update
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes action, log probability, entropy, and value for a batch of
        pre-processed states.

        Args:
            processed_state_batch (Dict[str, torch.Tensor]): Dictionary containing
                batched tensors for "position", "rotation_quat", "depth_map",
                "other_nodes_features", "other_nodes_mask".
            action (Optional[torch.Tensor]): If provided, computes properties for
                this specific action batch (used during RPO update). If None,
                samples a new action.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - action: The sampled or provided action [B, action_dim].
                - log_prob: Log probability of the action [B].
                - entropy: Entropy of the action distribution [B].
                - value: Value estimate for the state [B].
        """
        # Extract features from the pre-processed batch
        pos_batch = processed_state_batch["position"]
        rot_quat_batch = processed_state_batch["rotation_quat"]
        depth_batch = processed_state_batch["depth_map"]
        other_nodes_padded_batch = processed_state_batch["other_nodes_features"]
        other_nodes_mask_batch = processed_state_batch["other_nodes_mask"]

        # --- Feature Extraction and Combination ---
        # 1. Process inputs through respective modules
        vector_input = torch.cat([pos_batch, rot_quat_batch], dim=-1)
        vector_features = self.vector_processor(vector_input)
        cnn_features = self.cnn_processor(depth_batch)
        other_nodes_features = self._process_other_nodes(
            other_nodes_padded_batch, other_nodes_mask_batch
        )

        # 2. Combine features
        combined_features = torch.cat(
            [vector_features, cnn_features, other_nodes_features], dim=-1
        )
        combined_embedding = self.combiner(combined_features)  # Shape: [B, 256]

        # --- Actor-Critic Outputs ---
        # 3. Get raw action distribution mean and value estimate
        action_mean_raw = self.actor_mean(combined_embedding)
        value = self.critic(combined_embedding)  # Shape: [B, 1]

        # --- Action Distribution and Sampling ---
        # Assume action uses quaternion, normalize the mean's quaternion part
        action_mean_pos = action_mean_raw[..., :-QUATERNION_DIM]
        action_mean_quat_raw = action_mean_raw[..., -QUATERNION_DIM:]
        action_mean_quat = F.normalize(action_mean_quat_raw, p=2, dim=-1)
        action_mean = torch.cat([action_mean_pos, action_mean_quat], dim=-1)

        # Expand log std dev
        action_logstd_expanded = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd_expanded)
        # Create distribution using normalized mean
        probs = torch.distributions.Normal(action_mean, action_std)

        # Inject RPO noise if action is provided (during learning update)
        if action is not None:
            # Add noise to the *raw* mean before creating distribution for log_prob calculation
            # This matches the original RPO implementation's intent during updates
            z = (
                torch.FloatTensor(action_mean_raw.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(self.device)  # Ensure noise is on correct device
            )
            action_mean_noisy_raw = action_mean_raw + z
            probs_for_logprob = torch.distributions.Normal(
                action_mean_noisy_raw, action_std
            )
            # Use the *provided* action (from buffer) for log_prob calculation
            action_to_evaluate = action
        else:
            # Sample a new action if none provided
            action_sampled_raw = probs.sample()
            # Normalize the quaternion part of the sampled action
            action_pos = action_sampled_raw[..., :-QUATERNION_DIM]
            action_quat_raw = action_sampled_raw[..., -QUATERNION_DIM:]
            action_quat_normalized = F.normalize(action_quat_raw, p=2, dim=-1)
            action = torch.cat([action_pos, action_quat_normalized], dim=-1)
            # Use the distribution based on the normalized mean for log_prob when sampling
            probs_for_logprob = probs
            action_to_evaluate = action

        # Calculate log probability and entropy
        # Log prob calculation uses the distribution WITH noise if action was provided
        log_prob = probs_for_logprob.log_prob(action_to_evaluate).sum(
            1
        )  # Sum over action dimensions
        # Entropy depends only on std dev, so using original `probs` is fine
        entropy = probs.entropy().sum(1)  # Sum over action dimensions

        return action, log_prob, entropy, value.flatten()  # Return flattened value [B]
