# agent/models.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List

# Import model attributes
from .model_attributes import (
    PARTIAL_EMBEDDING_DIM,  # Output dim of each processor
    DEPTH_MAP_HEIGHT,  # Needed by CNNProcessor
    DEPTH_MAP_WIDTH,  # Needed by CNNProcessor
    TRANSFORMER_D_MODEL,  # Needed by OtherNodesProcessor
)

# Import utility functions
from .utils import layer_init

# Import processor modules
from .processors import VectorProcessor, CNNProcessor, OtherNodesProcessor


class Agent(nn.Module):
    """
    The main Agent network using a modular structure for input processing heads.
    Combines features from different processors and outputs policy and value estimates.
    """

    def __init__(self, action_space_shape: Tuple[int], rpo_alpha: float):
        """
        Initializes the Agent network with modular processors.

        Args:
            action_space_shape (Tuple[int]): Shape of the action space.
                                            Expected to be (POSITION_DIM + ROTATION_DIM,).
            rpo_alpha (float): Alpha parameter for RPO noise injection.
        """
        super().__init__()
        self.rpo_alpha = rpo_alpha
        action_dim = np.prod(action_space_shape).item()

        # --- 1. Input Processing Modules (Defined using nn.ModuleDict) ---
        # Keys in this dictionary ('vector', 'cnn', 'other_nodes') MUST match
        # the keys provided in the 'processed_state_batch' dictionary from the Trainer.
        self.processors = nn.ModuleDict(
            {
                "vector": VectorProcessor(output_dim=PARTIAL_EMBEDDING_DIM),
                "cnn": CNNProcessor(
                    h=DEPTH_MAP_HEIGHT,
                    w=DEPTH_MAP_WIDTH,
                    output_dim=PARTIAL_EMBEDDING_DIM,
                ),
                "other_nodes": OtherNodesProcessor(
                    d_model=TRANSFORMER_D_MODEL,  # Pass relevant args
                    output_dim=PARTIAL_EMBEDDING_DIM,
                ),
                # --- To add a new head (e.g., for audio): ---
                # 1. Define AudioProcessor in processors.py outputting PARTIAL_EMBEDDING_DIM
                # 2. Add it here:
                # 'audio': AudioProcessor(output_dim=PARTIAL_EMBEDDING_DIM),
                # 3. Ensure Trainer prepares 'audio' data in the state dict.
            }
        )

        # --- 2. Feature Combination ---
        # Calculate the combined dimension based on the number of processors and their output dim
        num_processors = len(self.processors)
        combined_embedding_dim = num_processors * PARTIAL_EMBEDDING_DIM
        print(f"Number of processor heads: {num_processors}")
        print(f"Combined embedding dimension: {combined_embedding_dim}")

        # Combiner network to merge features from all processor heads
        self.combiner = nn.Sequential(
            # Input size is now dynamically calculated
            layer_init(nn.Linear(combined_embedding_dim, 256)),
            nn.ReLU(),
            # Optional: Add more layers or LayerNorm here
        )

        # --- 3. Output Heads ---
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)  # Value head
        self.actor_mean = layer_init(
            nn.Linear(256, action_dim), std=0.01
        )  # Policy mean head
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, action_dim)
        )  # Policy log std dev

    def forward_features(
        self, processed_state_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Processes the input state dictionary through the respective processor modules
        and combines the resulting features.

        Args:
            processed_state_batch (Dict[str, torch.Tensor]): Dictionary containing
                batched tensors for each input modality. Keys must match the keys
                in self.processors (e.g., 'vector', 'cnn', 'other_nodes').
                The value for 'other_nodes' should be a tuple (features, mask).

        Returns:
            torch.Tensor: Combined feature embedding after passing through the combiner.
                          Shape [Batch, 256] (based on combiner output).
        """
        processed_features_list: List[
            torch.Tensor
        ] = []  # List to store outputs from each processor

        # Iterate through the processors defined in the ModuleDict
        for key, processor_module in self.processors.items():
            if key not in processed_state_batch:
                raise KeyError(
                    f"Input key '{key}' not found in processed_state_batch. "
                    f"Available keys: {list(processed_state_batch.keys())}"
                )

            input_data = processed_state_batch[key]

            # Handle tuple inputs (like for other_nodes processor)
            if isinstance(input_data, tuple):
                partial_embedding = processor_module(
                    *input_data
                )  # Unpack tuple as arguments
            else:
                partial_embedding = processor_module(input_data)  # Pass single tensor

            processed_features_list.append(partial_embedding)

        # Concatenate features from all processors along the feature dimension
        # Shape: [Batch, num_processors * PARTIAL_EMBEDDING_DIM]
        combined_features = torch.cat(processed_features_list, dim=-1)

        # Pass the combined features through the final combiner network
        # Shape: [Batch, 256]
        final_embedding = self.combiner(combined_features)

        return final_embedding

    def get_value(self, processed_state_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the value estimate V(s) for a batch of pre-processed states."""
        # Process inputs and combine features using the modular forward_features method
        combined_embedding = self.forward_features(processed_state_batch)
        # Get value estimate from the critic head
        value = self.critic(combined_embedding)
        return value  # Shape [B, 1]

    def get_action_and_value(
        self,
        processed_state_batch: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes action, log probability, entropy, and value estimate.
        """
        # Process inputs and combine features
        combined_embedding = self.forward_features(
            processed_state_batch
        )  # Shape: [B, 256]

        # --- Actor-Critic Outputs ---
        action_mean_raw = self.actor_mean(combined_embedding)  # Raw mean from network
        value = self.critic(combined_embedding)  # Value estimate [B, 1]

        # --- Action Distribution and Sampling (Gaussian, Z-Rotation Only) ---
        action_mean = (
            action_mean_raw  # No normalization needed for Pos + RotAngle output
        )

        action_logstd_expanded = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd_expanded)
        probs = torch.distributions.Normal(
            action_mean, action_std
        )  # Gaussian distribution

        # --- RPO Noise / Sampling Logic ---
        if (
            action is not None
        ):  # During training update: Use provided action, add noise to mean for logprob
            noise = (
                torch.FloatTensor(action_mean_raw.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(action_mean_raw.device)
            )
            action_mean_noisy_raw = action_mean_raw + noise
            probs_for_logprob = torch.distributions.Normal(
                action_mean_noisy_raw, action_std
            )
            action_to_evaluate = action
        else:  # During rollout: Sample action, use clean distribution for logprob
            action_sampled = probs.sample()
            action = action_sampled
            probs_for_logprob = probs
            action_to_evaluate = action

        # Calculate log probability and entropy
        log_prob = probs_for_logprob.log_prob(action_to_evaluate).sum(
            dim=1
        )  # Sum over action dim
        entropy = probs.entropy().sum(dim=1)  # Sum over action dim

        return action, log_prob, entropy, value.flatten()  # Return flattened value [B]
