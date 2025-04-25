import torch
import numpy as np
from .model_attributes import MAX_OTHER_NODES, OTHER_NODE_FEATURE_DIM


# Placeholder for layer_init function (assuming it exists as before)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _process_other_nodes(positions_list, rotations_quat_list, device):
    """
    Helper to process variable-length lists.
    Handles padding and processing (e.g., via Transformer).
    Input: List of position lists, List of rotation lists (one list per batch item)
    Output: Tensor of processed features (Batch, EmbedDim // 3)
    """

    batch_size = len(positions_list)
    padded_features = torch.zeros(
        batch_size, MAX_OTHER_NODES, OTHER_NODE_FEATURE_DIM, device=device
    )  # Updated feature dim
    masks = torch.zeros(batch_size, MAX_OTHER_NODES, dtype=torch.bool, device=device)

    for i in range(batch_size):
        num_nodes = min(len(positions_list[i]), MAX_OTHER_NODES)
        if num_nodes > 0:
            pos = torch.tensor(
                positions_list[i][:num_nodes], dtype=torch.float32, device=device
            )
            # Use quaternions directly
            quat = torch.stack(rotations_quat_list[i][:num_nodes]).to(
                device
            )  # Assuming rotations_quat_list contains tensors
            # quat = torch.tensor(rotations_quat_list[i][:num_nodes], dtype=torch.float32, device=device) # If list of lists/tuples

            features = torch.cat([pos, quat], dim=-1)  # Shape: (num_nodes, 7)
            padded_features[i, :num_nodes, :] = features
            masks[i, :num_nodes] = False
            if num_nodes < MAX_OTHER_NODES:
                masks[i, num_nodes:] = True

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
