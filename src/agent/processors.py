# agent/processors.py

import torch
import torch.nn as nn
from typing import Tuple
from .model_attributes import (
    VECTOR_INPUT_DIM,
    DEPTH_MAP_CHANNELS,
    DEPTH_MAP_HEIGHT,  # Needed for CNN calculation
    DEPTH_MAP_WIDTH,  # Needed for CNN calculation
    OTHER_NODE_RAW_FEATURE_DIM,
    PARTIAL_EMBEDDING_DIM,  # Target output dim for each processor
    TRANSFORMER_D_MODEL,  # Internal dimension for Transformer
)
from .utils import layer_init  # Assumes layer_init utility exists


# --- Processor for Agent's Own Vector Input (Position + Rotation Angle) ---
class VectorProcessor(nn.Module):
    """Processes the agent's own position and rotation angle."""

    def __init__(self, input_dim=VECTOR_INPUT_DIM, output_dim=PARTIAL_EMBEDDING_DIM):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, output_dim)),
            # Optional: Add another ReLU or LayerNorm here
            # nn.ReLU()
        )

    def forward(self, vector_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_input (torch.Tensor): Shape [Batch, VECTOR_INPUT_DIM]

        Returns:
            torch.Tensor: Shape [Batch, PARTIAL_EMBEDDING_DIM]
        """
        return self.network(vector_input)


# --- Processor for Depth Map Input ---
class CNNProcessor(nn.Module):
    """Processes the depth map image using a CNN."""

    def __init__(
        self,
        input_channels=DEPTH_MAP_CHANNELS,
        h=DEPTH_MAP_HEIGHT,
        w=DEPTH_MAP_WIDTH,  # Pass H/W for dynamic calculation
        output_dim=PARTIAL_EMBEDDING_DIM,
    ):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened size dynamically based on input H, W and CNN layers
        with torch.no_grad():
            # Create a dummy input tensor with the expected shape
            dummy_input = torch.zeros(1, input_channels, h, w)
            # Pass the dummy input through the CNN layers (excluding the final Linear layer)
            cnn_output_shape = self.network(dummy_input).shape
            # The second dimension of the output shape is the flattened size
            flattened_size = cnn_output_shape[1]
            print(f"CNN flattened size calculated as: {flattened_size}")

        # Add the final Linear layer using the calculated flattened size
        self.fc = layer_init(nn.Linear(flattened_size, output_dim))
        # Optional: Add another ReLU or LayerNorm here
        # self.output_activation = nn.ReLU()

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_map (torch.Tensor): Shape [Batch, C, H, W]

        Returns:
            torch.Tensor: Shape [Batch, PARTIAL_EMBEDDING_DIM]
        """
        features = self.network(depth_map)
        output = self.fc(features)
        # if hasattr(self, 'output_activation'):
        #     output = self.output_activation(output)
        return output


# --- Processor for Other Nodes Input (using Transformer) ---
class OtherNodesProcessor(nn.Module):
    """
    Processes features of other nearby nodes using projection, Transformer,
    and aggregation.
    """

    def __init__(
        self,
        input_feature_dim=OTHER_NODE_RAW_FEATURE_DIM,
        d_model=TRANSFORMER_D_MODEL,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward_factor=2,
        dropout=0.1,
        output_dim=PARTIAL_EMBEDDING_DIM,
    ):
        super().__init__()

        # 1. Input Projection Layer
        # Projects raw node features (Pos+RotAngle) to the Transformer's model dimension (d_model)
        self.input_projection = nn.Sequential(
            layer_init(nn.Linear(input_feature_dim, d_model)),
            nn.ReLU(),
            # Optional: LayerNorm(d_model)
        )

        # 2. Transformer Encoder Layer(s)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * dim_feedforward_factor,
            dropout=dropout,
            batch_first=True,  # Expects input shape (Batch, SeqLen, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 3. Output Aggregation and Projection Layer
        # Processes the aggregated Transformer output to the final partial embedding dimension
        self.output_projection = nn.Sequential(
            layer_init(nn.Linear(d_model, output_dim))
            # Optional: nn.ReLU() or nn.LayerNorm(output_dim)
        )

    def forward(
        self, other_nodes_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            other_nodes_data (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - features (torch.Tensor): Padded features of other nodes.
                                           Shape [Batch, MAX_OTHER_NODES, OTHER_NODE_RAW_FEATURE_DIM]
                - mask (torch.Tensor): Boolean padding mask (True for padding).
                                       Shape [Batch, MAX_OTHER_NODES]

        Returns:
            torch.Tensor: Processed embedding for other nodes. Shape [Batch, PARTIAL_EMBEDDING_DIM]
        """
        features, mask = other_nodes_data  # Unpack the input tuple

        # 1. Project input features
        # Input: [B, SeqLen, RawFeatDim] -> Output: [B, SeqLen, d_model]
        projected_features = self.input_projection(features)

        # 2. Apply Transformer Encoder
        # src_key_padding_mask needs shape [B, SeqLen] where True indicates padding
        # Output: [B, SeqLen, d_model]
        transformer_output = self.transformer_encoder(
            projected_features, src_key_padding_mask=mask
        )

        # 3. Aggregate Transformer output (masked mean pooling)
        # Zero out embeddings corresponding to padding tokens before summing
        transformer_output_masked = transformer_output.masked_fill(
            mask.unsqueeze(-1),
            0.0,  # Unsqueeze mask to broadcast over feature dim
        )
        # Count non-padded elements per batch item for accurate averaging
        non_pad_count = (
            (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        )  # Avoid division by zero
        # Sum features across the sequence dimension (SeqLen = MAX_OTHER_NODES)
        summed_features = transformer_output_masked.sum(dim=1)  # Shape: [B, d_model]
        # Calculate the mean by dividing by the count of valid tokens
        mean_features = summed_features / non_pad_count  # Shape: [B, d_model]

        # 4. Final output projection
        # Input: [B, d_model] -> Output: [B, PARTIAL_EMBEDDING_DIM]
        processed_embedding = self.output_projection(mean_features)

        return processed_embedding
