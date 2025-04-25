import torch.nn as nn
from .model_attributes import (
    VECTOR_INPUT_DIM,
    EMBEDDING_DIM,
    DEPTH_MAP_CHANNELS,
    OTHER_NODE_FEATURE_DIM,
    HEADCOUNT,
)
from .utils import layer_init

vector_processor = nn.Sequential(
    layer_init(nn.Linear(VECTOR_INPUT_DIM, 64)),  # Updated input size
    nn.ReLU(),
    layer_init(nn.Linear(64, EMBEDDING_DIM // HEADCOUNT)),
)

cnn_processor = nn.Sequential(
    layer_init(nn.Conv2d(DEPTH_MAP_CHANNELS, 32, kernel_size=8, stride=4)),
    nn.ReLU(),
    layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
    nn.ReLU(),
    layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
    nn.ReLU(),
    nn.Flatten(),
    # Calculate the flattened size based on CNN output and H, W
    # Example calculation for 64x64 input:
    # Conv1: (64 - 8) / 4 + 1 = 15
    # Conv2: (15 - 4) / 2 + 1 = 6.5 -> 6 (floor)
    # Conv3: (6 - 3) / 1 + 1 = 4
    # Flattened size: 64 * 4 * 4 = 1024
    layer_init(
        nn.Linear(64 * 4 * 4, EMBEDDING_DIM // HEADCOUNT)
    ),  # Output partial embedding
)
other_nodes_processor = nn.Sequential(
    # Project each node's features (pos+rot) to embedding dim
    layer_init(nn.Linear(OTHER_NODE_FEATURE_DIM, EMBEDDING_DIM)),
    nn.ReLU(),
    # Note: Transformer layer expects input shape (SeqLen, Batch, EmbedDim)
    # We will need to permute dimensions before passing to the encoder layer
)
