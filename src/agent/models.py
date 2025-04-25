import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.tv_tensors import Image  # Assuming this is torch.Tensor compatible
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass


# Placeholder for layer_init function (assuming it exists as before)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Define expected dimensions (replace with your actual values)
DEPTH_MAP_CHANNELS = 1
DEPTH_MAP_HEIGHT = 64
DEPTH_MAP_WIDTH = 64
MAX_OTHER_NODES = 10
QUATERNION_DIM = 4
POSITION_DIM = 3
VECTOR_INPUT_DIM = QUATERNION_DIM + POSITION_DIM  # Now 7
OTHER_NODE_FEATURE_DIM = QUATERNION_DIM + POSITION_DIM  # Now 7
EMBEDDING_DIM = 128  # Example dimension for processed features


@dataclass
class SingleNodeState:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    depth_map: Image  # Should behave like a Tensor [C, H, W]
    position_of_other_nodes: list[tuple[float, float, float]]
    rotation_of_other_nodes: list[tuple[float, float, float]]


class Agent(nn.Module):
    def __init__(self, action_space_shape, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        action_dim = np.prod(action_space_shape)

        # --- 1. Input Processing Modules ---

        # Module for own position and rotation (simple vectors)
        self.vector_processor = nn.Sequential(
            layer_init(nn.Linear(VECTOR_INPUT_DIM, 64)),  # Updated input size
            nn.ReLU(),
            layer_init(nn.Linear(64, EMBEDDING_DIM // 3)),
        )

        # Module for depth map (CNN) - Define your own CNN architecture
        self.cnn_processor = nn.Sequential(
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
                nn.Linear(64 * 4 * 4, EMBEDDING_DIM // 3)
            ),  # Output partial embedding
        )

        # Module for variable-length lists of other nodes
        # Option A: Padding + Transformer Encoder (Powerful, handles relations)
        self.other_nodes_processor = nn.Sequential(
            # Project each node's features (pos+rot) to embedding dim
            layer_init(nn.Linear(OTHER_NODE_FEATURE_DIM, EMBEDDING_DIM)),
            nn.ReLU(),
            # Note: Transformer layer expects input shape (SeqLen, Batch, EmbedDim)
            # We will need to permute dimensions before passing to the encoder layer
        )
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
                nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // 3)
            )  # Output partial embedding
        )

        # Option B: Simpler Aggregation (Mean/Max Pooling) - Uncomment if preferred
        # self.other_nodes_processor = nn.Sequential(
        #     layer_init(nn.Linear(OTHER_NODE_FEATURE_DIM, 64)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(64, EMBEDDING_DIM // 3))
        # )
        # # Aggregation (e.g., mean) will happen in the forward pass

        # --- 2. Feature Combination ---
        combined_embedding_dim = (
            EMBEDDING_DIM // 3
        ) * 3  # Sum of partial embedding dims
        self.combiner = nn.Sequential(
            layer_init(nn.Linear(combined_embedding_dim, 256)), nn.ReLU()
        )

        # --- 3. Output Heads ---
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def _process_other_nodes(self, positions_list, rotations_quat_list, device):
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
        masks = torch.zeros(
            batch_size, MAX_OTHER_NODES, dtype=torch.bool, device=device
        )

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

        # Option B: Aggregation (Simpler) - Uncomment if using this
        # if padded_features.numel() == 0 or masks.all(): # Handle case with no valid nodes
        #     return torch.zeros(batch_size, EMBEDDING_DIM // 3, device=device)
        # x = self.other_nodes_processor(padded_features) # (B, SeqLen, EmbedDim//3)
        # # Masked mean pooling
        # x = x.masked_fill(masks.unsqueeze(-1), 0.0) # Zero out padding embeddings
        # non_pad_count = (~masks).sum(dim=1, keepdim=True) # Count valid tokens (B, 1)
        # summed_features = x.sum(dim=1) # (B, EmbedDim//3)
        # # Avoid division by zero
        # processed_other_nodes = torch.where(non_pad_count > 0, summed_features / non_pad_count, torch.zeros_like(summed_features))

        return processed_other_nodes

    def get_value(self, state: SingleNodeState):
        # This function now needs to process the structured state
        # It's easier to just call get_action_and_value and extract the value
        _, _, _, value = self.get_action_and_value(state)
        return value

    def get_action_and_value(self, state_batch, action=None):
        # --- Batch Handling & Euler->Quat Conversion ---
        device = self.actor_logstd.device
        # Assume state_batch provides Euler rotations initially
        # Convert Euler to Quat before passing to internal processing
        # This requires knowing the Euler sequence (e.g., 'xyz', 'zyx')
        euler_seq = 'xyz' # IMPORTANT: Match environment's convention

        # Example: Based on previous structure (list of SingleNodeState)
        if isinstance(state_batch, list) and all(isinstance(s, SingleNodeState) for s in state_batch):
            # Own rotation conversion
            own_rot_euler_list = [s.rotation for s in state_batch]
            own_rot_quat_scipy = R.from_euler(euler_seq, own_rot_euler_list, degrees=False).as_quat() # Scipy quat: (x,y,z,w)
            # Convert Scipy (x,y,z,w) to PyTorch Tensor (w,x,y,z) or (x,y,z,w) - be consistent! Let's use (w,x,y,z)
            own_rot_quat = torch.tensor(np.roll(own_rot_quat_scipy, 1, axis=1), dtype=torch.float32, device=device) # Now (w,x,y,z)

            # Other nodes rotation conversion
            other_pos_list = [s.position_of_other_nodes for s in state_batch]
            other_rot_quat_list = []
            for rot_list_euler in [s.rotation_of_other_nodes for s in state_batch]:
                 if rot_list_euler:
                      quats_scipy = R.from_euler(euler_seq, rot_list_euler, degrees=False).as_quat()
                      quats_torch = torch.tensor(np.roll(quats_scipy, 1, axis=1), dtype=torch.float32) # List of Tensors (w,x,y,z)
                      other_rot_quat_list.append(list(quats_torch)) # Store list of tensors
                 else:
                      other_rot_quat_list.append([])

            # Extract other components
            pos_batch = torch.tensor([s.position for s in state_batch], dtype=torch.float32, device=device)
            depth_batch = torch.stack([s.depth_map for s in state_batch]).to(device)

        elif isinstance(state_batch, dict):
            # Handle dict-based batching similarly, converting Euler rotations
            # pos_batch = state_batch['position'].to(device)
            # own_rot_euler = state_batch['rotation'].cpu().numpy() # Example if tensor
            # own_rot_quat_scipy = R.from_euler(euler_seq, own_rot_euler, degrees=False).as_quat()
            # own_rot_quat = torch.tensor(np.roll(own_rot_quat_scipy, 1, axis=1), dtype=torch.float32, device=device)
            # ... etc ...
            raise NotImplementedError("Dict batching conversion not fully implemented")
        else:
            raise TypeError(f"Unsupported state_batch type: {type(state_batch)}")
        # --- End Batch Handling ---


        # 1. Process inputs (Now using quaternions internally)
        vector_input = torch.cat([pos_batch, own_rot_quat], dim=-1) # Use converted quat
        vector_features = self.vector_processor(vector_input)
        cnn_features = self.cnn_processor(depth_batch)
        # Pass list of quaternion lists to helper
        other_nodes_features = self._process_other_nodes(other_pos_list, other_rot_quat_list, device)

        # 2. Combine features
        combined_features = torch.cat([vector_features, cnn_features, other_nodes_features], dim=-1)
        combined_embedding = self.combiner(combined_features)

        # 3. Get action distribution mean and value estimate
        action_mean_raw = self.actor_mean(combined_embedding)
        value = self.critic(combined_embedding)

        # --- Action processing ---
        # Assume action_mean_raw contains quaternion components (e.g., last 4 dims)
        # Example: action = [dx, dy, dz, qw, qx, qy, qz]
        action_mean_pos = action_mean_raw[..., :-QUATERNION_DIM] # Position/other parts
        action_mean_quat_raw = action_mean_raw[..., -QUATERNION_DIM:] # Raw quaternion part

        # Normalize the quaternion part of the mean for the distribution
        action_mean_quat = F.normalize(action_mean_quat_raw, p=2, dim=-1)
        action_mean = torch.cat([action_mean_pos, action_mean_quat], dim=-1)

        # --- Action sampling / LogProb Calculation ---
        action_logstd_expanded = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd_expanded)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action_sampled_raw = probs.sample()
            # Normalize the sampled quaternion part again
            action_pos = action_sampled_raw[..., :-QUATERNION_DIM]
            action_quat_raw = action_sampled_raw[..., -QUATERNION_DIM:]
            action_quat_normalized = F.normalize(action_quat_raw, p=2, dim=-1)
            action = torch.cat([action_pos, action_quat_normalized], dim=-1)
        # else: action is provided (during update), assume it's already normalized if necessary

        # Calculate log_prob using the potentially unnormalized mean for stability,
        # but evaluate at the normalized (or provided) action.
        # Recalculate distribution with raw mean for log_prob
        probs_for_logprob = torch.distributions.Normal(action_mean_raw, action_std)
        log_prob = probs_for_logprob.log_prob(action).sum(1) # Sum over action dimensions
        # Entropy depends only on std deviation for Normal dist, so it's okay
        entropy = probs.entropy().sum(1) # Sum over action dimensions

        return action, log_prob, entropy, value.flatten()


# Example usage (conceptual, assuming env and batching are handled)
# action_dim = ... # Define based on your environment's action space
# agent = Agent(action_space_shape=(action_dim,), rpo_alpha=args.rpo_alpha).to(device)

# During training loop (after collecting data and creating a minibatch 'mb_state_batch')
# mb_state_batch = ... # Your custom batch object/dict/list for the minibatch
# actions_in_batch = ... # Corresponding actions from the batch
# _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_state_batch, actions_in_batch)
