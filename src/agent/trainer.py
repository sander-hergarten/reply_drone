# agent/trainer.py

import torch
import torch.optim as optim
import numpy as np
import random
import time
import toml
from types import SimpleNamespace
from os import PathLike
from typing import List, Dict, Any

# Import necessary components
from .models import Agent  # Agent model with modular processors
from .types import SingleNodeState, SingleAction
from .utils import (
    filter_by_n_closest,
    _batch_extract_z_rotation,
    _batch_pad_other_nodes,
    # layer_init # Not directly used in trainer
)
from .model_attributes import (
    MAX_OTHER_NODES,
    # OTHER_NODE_FEATURE_DIM, # Renamed/replaced usage
    OTHER_NODE_RAW_FEATURE_DIM,  # Use raw feature dim for padding function
    ROTATION_DIM,
    POSITION_DIM,
    # PARTIAL_EMBEDDING_DIM # Not directly used in trainer
)

# Optional logging imports
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    """
    Handles ON-POLICY RPO training with a modular Agent structure.
    Collects rollouts, computes GAE, and updates the agent.
    Assumes Z-axis rotation only.
    """

    # __init__ remains largely the same, ensure hyperparameters are loaded
    def __init__(self, hyperparameters: PathLike, environment: Any):
        self.args = SimpleNamespace(**toml.load(hyperparameters))
        self.environment = environment

        # --- Validate required hyperparameters ---
        required_hparams = [
            "env_id",
            "exp_name",
            "seed",
            "cuda",
            "track",
            "learning_rate",
            "rpo_alpha",
            "num_steps",
            "num_minibatches",
            "update_epochs",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "vf_coef",
            "ent_coef",
            "total_timesteps",
        ]
        for param in required_hparams:
            if not hasattr(self.args, param):
                raise AttributeError(
                    f"Hyperparameter '{param}' is missing in {hyperparameters}."
                )

        # Use the actual number of environments
        if (
            hasattr(self.args, "num_envs")
            and self.args.num_envs != self.environment.num_envs
        ):
            print(
                f"Warning: Overriding args.num_envs ({self.args.num_envs}) "
                f"with environment.num_envs ({self.environment.num_envs})"
            )
        self.args.num_envs = self.environment.num_envs

        # --- Calculate runtime arguments ---
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        if self.args.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive.")
        self.args.minibatch_size = int(
            self.args.batch_size // self.args.num_minibatches
        )
        if self.args.minibatch_size == 0 and self.args.batch_size > 0:
            print(
                f"Warning: minibatch_size is zero (batch_size={self.args.batch_size}, num_minibatches={self.args.num_minibatches}). Setting num_minibatches to 1."
            )
            self.args.num_minibatches = 1
            self.args.minibatch_size = self.args.batch_size
        elif self.args.minibatch_size == 0:
            raise ValueError("batch_size is zero, cannot train.")

        self.args.num_updates = self.args.total_timesteps // self.args.batch_size
        if self.args.num_updates == 0:
            raise ValueError(
                f"total_timesteps ({self.args.total_timesteps}) is less than batch_size ({self.args.batch_size}). No updates possible."
            )

        # --- Set defaults for optional hyperparameters ---
        self.args.wandb_project_name = getattr(
            self.args, "wandb_project_name", "RPO_Trainer_Modular"
        )
        self.args.wandb_entity = getattr(self.args, "wandb_entity", None)
        self.args.torch_deterministic = getattr(self.args, "torch_deterministic", True)
        self.args.norm_adv = getattr(self.args, "norm_adv", True)
        self.args.clip_vloss = getattr(self.args, "clip_vloss", True)
        self.args.max_grad_norm = getattr(self.args, "max_grad_norm", 0.5)
        self.args.target_kl = getattr(self.args, "target_kl", None)
        self.args.anneal_lr = getattr(self.args, "anneal_lr", True)

        # --- Initialize internal state ---
        self.run_name = None
        self.device = None
        self.agent: Agent = None
        self.optimizer = None
        self.writer: SummaryWriter = None
        self.global_step = 0

        # --- Process Euler sequence ---
        self.euler_seq = getattr(self.args, "euler_seq", "xyz").lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if len(self.euler_seq) != 3 or not all(c in axis_map for c in self.euler_seq):
            raise ValueError(
                f"Invalid euler_seq: '{self.euler_seq}'. Must be a 3-character permutation of 'x', 'y', 'z'."
            )
        self.euler_indices = tuple(axis_map[c] for c in self.euler_seq)
        self.z_axis_index_in_euler = self.euler_indices[2]

        # Agent's action dimension
        self.agent_action_dim = POSITION_DIM + ROTATION_DIM

    # setup remains the same as before
    def setup(self):
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        # --- Logging Setup ---
        if self.args.track and wandb:
            try:
                wandb.init(
                    project=self.args.wandb_project_name,
                    entity=self.args.wandb_entity,
                    sync_tensorboard=True,
                    config=vars(self.args),
                    name=self.run_name,
                    save_code=True,
                )
                print("Weights & Biases tracking enabled.")
            except Exception as e:
                print(f"WandB initialization failed: {e}. Disabling tracking.")
                self.args.track = False
        else:
            self.args.track = False
        if SummaryWriter:
            try:
                log_dir = f"runs/{self.run_name}"
                self.writer = SummaryWriter(log_dir)
                hparam_text = "|param|value|\n|-|-|\n%s" % (
                    "\n".join(
                        [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                    )
                )
                self.writer.add_text("hyperparameters", hparam_text)
                print(f"TensorBoard logging initialized at {log_dir}")
            except Exception as e:
                print(f"TensorBoard initialization failed: {e}. Logging disabled.")
                self.writer = None
        else:
            self.writer = None
        # --- Seeding ---
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        # --- Device Setup ---
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.args.cuda else "cpu"
        )
        print(f"Using device: {self.device}")
        # --- Agent and Optimizer Setup ---
        print(f"Agent action dimension set to: {self.agent_action_dim}")
        self.agent = Agent(
            action_space_shape=(self.agent_action_dim,), rpo_alpha=self.args.rpo_alpha
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )
        print("Agent and Optimizer initialized.")
        print("Setup complete.")

    # --- MODIFIED: _prepare_batch_states_for_agent ---
    def _prepare_batch_states_for_agent(
        self,
        states_list: List[SingleNodeState],
    ) -> Dict[str, Any]:  # Return type includes tuples now
        """
        Processes environment states into a dictionary of batched tensors,
        with keys matching the Agent's processor ModuleDict.

        Args:
            states_list: List of SingleNodeState objects from the environment.

        Returns:
            A dictionary where keys match Agent.processors keys (e.g., 'vector', 'cnn', 'other_nodes')
            and values are corresponding batched tensors or tuples of tensors.
        """
        batch_size = len(states_list)
        if batch_size == 0:
            return {}

        # --- Filter closest nodes ---
        filtered_pos_list = []
        filtered_rot_eul_list = []
        for i in range(batch_size):
            state = states_list[i]
            # Ensure position_of_other_nodes exists and is a dict
            pos_others = getattr(state, "position_of_other_nodes", {})
            rot_others = getattr(state, "rotation_of_other_nodes", {})
            if not isinstance(pos_others, dict):
                pos_others = {}
            if not isinstance(rot_others, dict):
                rot_others = {}

            f_pos, f_rot_eul = filter_by_n_closest(
                state.position,
                pos_others,
                rot_others,
                n=MAX_OTHER_NODES,
            )
            filtered_pos_list.append(f_pos)
            filtered_rot_eul_list.append(f_rot_eul)

        # --- Extract Z-rotation angles ---
        own_rot_eul_list = [s.rotation for s in states_list]
        own_rot_z_angle_batch, other_rot_z_angle_dict_list = _batch_extract_z_rotation(
            own_rot_eul_list, filtered_rot_eul_list, self.euler_indices, self.device
        )  # Shapes [B, 1] and List[Dict[id -> Tensor[1]]]

        # --- Pad other node features (Pos + Z-Angle) ---
        # Note: Pass OTHER_NODE_RAW_FEATURE_DIM to the padding function
        other_nodes_padded_batch, other_nodes_mask_batch = _batch_pad_other_nodes(
            filtered_pos_list,
            other_rot_z_angle_dict_list,
            max_nodes=MAX_OTHER_NODES,
            feature_dim=OTHER_NODE_RAW_FEATURE_DIM,  # Use raw dim here
            device=self.device,
        )  # Shapes [B, max_nodes, raw_feat_dim] and [B, max_nodes]

        # --- Prepare Tensors for Dictionary ---
        pos_batch = torch.tensor(
            [s.position for s in states_list], dtype=torch.float32, device=self.device
        )  # Shape [B, POSITION_DIM]

        # Combine own position and rotation for the 'vector' processor input
        vector_input_batch = torch.cat(
            [pos_batch, own_rot_z_angle_batch], dim=-1
        )  # Shape [B, Pos+Rot]

        # Stack depth maps
        depth_batch = torch.stack([s.depth_map for s in states_list]).to(self.device)
        if depth_batch.ndim == 3:
            depth_batch = depth_batch.unsqueeze(1)  # Ensure [B, C, H, W]

        # --- Create the final dictionary with keys matching Agent.processors ---
        processed_dict = {
            "vector": vector_input_batch,  # Input for VectorProcessor
            "cnn": depth_batch,  # Input for CNNProcessor
            # Input for OtherNodesProcessor is a tuple (features, mask)
            "other_nodes": (other_nodes_padded_batch, other_nodes_mask_batch),
            # --- To add a new head (e.g., 'audio'): ---
            # 1. Process audio data into a batch tensor: audio_batch = ...
            # 2. Add it here:
            # 'audio': audio_batch,
        }

        return processed_dict

    # --- END MODIFIED ---

    # _convert_action_tensor_to_env_dict remains the same as before
    def _convert_action_tensor_to_env_dict(
        self, action_tensor: torch.Tensor
    ) -> Dict[int, SingleAction]:
        action_np = action_tensor.cpu().numpy()
        num_envs = action_np.shape[0]
        action_pos_part = action_np[:, :POSITION_DIM]
        action_z_angle_part = action_np[:, POSITION_DIM:]
        action_rot_euler = np.zeros((num_envs, 3), dtype=np.float32)
        action_rot_euler[:, self.z_axis_index_in_euler] = action_z_angle_part.flatten()
        action_dict = {}
        for i in range(num_envs):
            action_dict[i] = SingleAction(
                id=i,
                new_pos=tuple(action_pos_part[i]),
                new_rot=tuple(action_rot_euler[i]),
            )
        return action_dict

    # _convert_env_obs_dict_to_state_list remains the same
    def _convert_env_obs_dict_to_state_list(
        self, obs_dict: Dict[int, SingleNodeState]
    ) -> List[SingleNodeState]:
        num_envs = len(obs_dict)
        state_list = [obs_dict[i] for i in range(num_envs)]
        return state_list

    # start_training needs minor adjustments to handle the processed state dict correctly
    def start_training(self):
        if self.agent is None or self.optimizer is None:
            raise RuntimeError("Trainer setup not complete. Call setup() first.")
        print("Starting training...")
        start_time = time.time()

        # --- Initial Reset ---
        initial_obs_dict = self.environment.reset(seed=self.args.seed)
        next_states_list = self._convert_env_obs_dict_to_state_list(initial_obs_dict)
        # Prepare initial state dict for the agent
        next_processed_states_dict = self._prepare_batch_states_for_agent(
            next_states_list
        )
        next_dones_tensor = torch.zeros(self.args.num_envs, device=self.device)

        # --- Main Training Loop ---
        print(f"Total updates: {self.args.num_updates}")
        for update in range(1, self.args.num_updates + 1):
            # --- LR Annealing ---
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.args.num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # --- 1. Rollout Phase ---
            # Store processed states directly
            rollout_processed_states = [None] * self.args.num_steps  # List[Dict]
            rollout_actions = torch.zeros(
                (self.args.num_steps, self.args.num_envs, self.agent_action_dim),
                device=self.device,
            )
            rollout_log_probs = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            rollout_rewards = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            rollout_dones = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            rollout_values = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            current_episode_returns = []
            current_episode_lengths = []

            for step in range(0, self.args.num_steps):
                self.global_step += self.args.num_envs
                # Store the processed state dict for this step
                rollout_processed_states[step] = (
                    next_processed_states_dict  # Store the dict
                )
                rollout_dones[step] = next_dones_tensor

                # Get action and value from agent using the processed dict
                with torch.no_grad():
                    action_agent_tensor, log_prob, _, value = (
                        self.agent.get_action_and_value(
                            next_processed_states_dict,
                            action=None,
                        )
                    )
                    rollout_values[step] = value.flatten()

                rollout_actions[step] = action_agent_tensor
                rollout_log_probs[step] = log_prob

                # Convert action and step environment
                action_dict_for_env = self._convert_action_tensor_to_env_dict(
                    action_agent_tensor
                )
                (
                    next_obs_dict,
                    rewards_dict,
                    terminated_dict,
                    truncated_dict,
                    infos_dict,
                ) = self.environment.step(action_dict_for_env)

                # Process rewards and dones
                rewards_tensor = torch.tensor(
                    [rewards_dict[i] for i in range(self.args.num_envs)],
                    dtype=torch.float32,
                    device=self.device,
                ).view(-1)
                terminated_tensor = torch.tensor(
                    [terminated_dict[i] for i in range(self.args.num_envs)],
                    dtype=torch.bool,
                    device=self.device,
                )
                truncated_tensor = torch.tensor(
                    [truncated_dict[i] for i in range(self.args.num_envs)],
                    dtype=torch.bool,
                    device=self.device,
                )
                dones_tensor = torch.logical_or(terminated_tensor, truncated_tensor)
                rollout_rewards[step] = rewards_tensor

                # Prepare next state dict
                next_states_list = self._convert_env_obs_dict_to_state_list(
                    next_obs_dict
                )
                next_processed_states_dict = self._prepare_batch_states_for_agent(
                    next_states_list
                )
                next_dones_tensor = dones_tensor.float()

                # Log episodic info (same as before)
                for env_id in range(self.args.num_envs):
                    if infos_dict and env_id in infos_dict and infos_dict[env_id]:
                        info = infos_dict[env_id]
                        final_info = info.get("final_info")
                        if not final_info and "episode" in info:
                            final_info = info
                        if (
                            final_info
                            and isinstance(final_info, dict)
                            and "episode" in final_info
                        ):
                            ep_info = final_info["episode"]
                            ep_return = ep_info.get("r")
                            ep_length = ep_info.get("l")
                            if ep_return is not None and ep_length is not None:
                                current_episode_returns.append(ep_return)
                                current_episode_lengths.append(ep_length)
                                if self.writer:
                                    self.writer.add_scalar(
                                        f"charts/episodic_return_env{env_id}",
                                        ep_return,
                                        self.global_step,
                                    )
                                    self.writer.add_scalar(
                                        f"charts/episodic_length_env{env_id}",
                                        ep_length,
                                        self.global_step,
                                    )

            # --- 2. GAE Calculation ---
            with torch.no_grad():
                # Use the last processed state dict for next_value calculation
                next_value = self.agent.get_value(next_processed_states_dict).reshape(
                    1, -1
                )
                advantages = torch.zeros_like(rollout_rewards).to(self.device)
                last_gae_lam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        next_nonterminal = 1.0 - next_dones_tensor
                        next_values_t = next_value
                    else:
                        next_nonterminal = 1.0 - rollout_dones[t + 1]
                        next_values_t = rollout_values[t + 1]
                    delta = (
                        rollout_rewards[t]
                        + self.args.gamma * next_values_t * next_nonterminal
                        - rollout_values[t]
                    )
                    advantages[t] = last_gae_lam = (
                        delta
                        + self.args.gamma
                        * self.args.gae_lambda
                        * next_nonterminal
                        * last_gae_lam
                    )
                returns = advantages + rollout_values

            # --- 3. Prepare Flattened Data ---
            # Collate the list of processed state dicts into a single batch dict
            b_states_processed = {}
            if rollout_processed_states and rollout_processed_states[0]:
                keys = rollout_processed_states[0].keys()
                for k in keys:
                    # Special handling for tuples (like 'other_nodes')
                    if isinstance(rollout_processed_states[0][k], tuple):
                        # Stack each element of the tuple separately
                        stacked_tuple_elements = []
                        for i in range(len(rollout_processed_states[0][k])):
                            stacked_el = torch.stack(
                                [
                                    step_data[k][i]
                                    for step_data in rollout_processed_states
                                ],
                                dim=0,
                            )
                            stacked_tuple_elements.append(
                                stacked_el.reshape(-1, *stacked_el.shape[2:])
                            )
                        b_states_processed[k] = tuple(stacked_tuple_elements)
                    else:  # Handle single tensors
                        stacked_tensor = torch.stack(
                            [step_data[k] for step_data in rollout_processed_states],
                            dim=0,
                        )
                        b_states_processed[k] = stacked_tensor.reshape(
                            -1, *stacked_tensor.shape[2:]
                        )
            else:
                print("Warning: rollout_processed_states is empty. Skipping update.")
                continue

            # Flatten other rollout data
            b_actions = rollout_actions.reshape(-1, self.agent_action_dim)
            b_old_log_probs = rollout_log_probs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_old_values = rollout_values.reshape(-1)

            # Advantage Normalization
            if self.args.norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

            # --- 4. Optimization Phase ---
            indices = np.arange(self.args.batch_size)
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_indices = indices[start:end]

                    # Get minibatch data from the collated batch dictionary
                    mb_states_processed = {}
                    for k, v in b_states_processed.items():
                        if isinstance(v, tuple):  # Handle tuple data
                            mb_states_processed[k] = tuple(el[mb_indices] for el in v)
                        else:  # Handle tensor data
                            mb_states_processed[k] = v[mb_indices]

                    mb_actions = b_actions[mb_indices]
                    mb_old_log_probs = b_old_log_probs[mb_indices]
                    mb_advantages = b_advantages[mb_indices]
                    mb_returns = b_returns[mb_indices]
                    mb_old_values = b_old_values[mb_indices]

                    # Re-evaluate actions and values for the minibatch
                    _, new_log_prob, entropy, new_value = (
                        self.agent.get_action_and_value(
                            mb_states_processed,  # Pass the minibatch processed dict
                            mb_actions,  # Pass actions for RPO noise
                        )
                    )

                    # PPO Loss Calculation (same as before)
                    log_ratio = new_log_prob - mb_old_log_probs
                    ratio = torch.exp(log_ratio)
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    new_value = new_value.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (new_value - mb_returns) ** 2
                        v_clipped = mb_old_values + torch.clamp(
                            new_value - mb_old_values,
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.args.ent_coef * entropy_loss
                        + self.args.vf_coef * v_loss
                    )

                    # Optimization Step (same as before)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.agent.parameters(), self.args.max_grad_norm
                        )
                    self.optimizer.step()

                # KL Early Stopping (same as before)
                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    print(
                        f"  Early stopping at epoch {epoch+1}/{self.args.update_epochs} due to reaching max KL divergence ({approx_kl:.4f} > {self.args.target_kl:.4f})."
                    )
                    break

            # --- 5. Logging ---
            if self.writer:
                y_pred, y_true = b_old_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
                self.writer.add_scalar(
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step,
                )
                self.writer.add_scalar(
                    "losses/value_loss", v_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/policy_loss", pg_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/entropy", entropy_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/approx_kl", approx_kl.item(), self.global_step
                )
                self.writer.add_scalar(
                    "charts/explained_variance", explained_var, self.global_step
                )
                if current_episode_returns:
                    self.writer.add_scalar(
                        "charts/mean_episodic_return",
                        np.mean(current_episode_returns),
                        self.global_step,
                    )
                if current_episode_lengths:
                    self.writer.add_scalar(
                        "charts/mean_episodic_length",
                        np.mean(current_episode_lengths),
                        self.global_step,
                    )

            # Print progress
            current_time = time.time()
            sps = int(self.args.batch_size / (current_time - start_time))
            print(
                f"Update {update}/{self.args.num_updates}, Timestep {self.global_step}/{self.args.total_timesteps}, SPS: {sps}"
            )
            start_time = current_time

        # --- End of Training ---
        print("Training finished.")
        self.close()

    # close remains the same
    def close(self):
        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")
        if self.args.track and wandb and wandb.run:
            wandb.finish()
            print("WandB run finished.")
