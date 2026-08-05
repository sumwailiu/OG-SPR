from collections import deque
from typing import Union, Tuple, Any
import numpy as np
import torch


# The replay buffer is build on https://github.com/facebookresearch/MRQ/blob/main/MRQ/buffer.py
class PriorityReplayBuffer:
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        pixel_obs: bool,
        device: torch.device,
        history: int = 1,
        horizon: int = 1,
        max_size: int = int(1e6),
        batch_size: int = 256,
        prioritized: bool = True,
        initial_priority: float = 1.0,
    ):
        self.device = device
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)

        self.obs_shape = obs_shape
        self.obs_dtype = np.uint8 if pixel_obs else np.float32

        self.state_shape = [obs_shape[0] * history] # Channels or obs dim.
        if pixel_obs:
            self.state_shape += [obs_shape[1], obs_shape[2]]    # Image size.
        self.num_channels = obs_shape[0]    # Used to grab only the most recent obs (history) or channels.

        # tracking
        self.ind, self.size = 0, 0
        self.ep_timesteps = 0
        self.env_terminates = False

        # History (used even if history = 1)
        self.history = int(history)
        self.state_ind = np.zeros((self.max_size, self.history), dtype=np.int32)
        self.next_ind = np.zeros((self.max_size, self.history), dtype=np.int32)

        self.history_queue = deque(maxlen=self.history)
        for _ in range(self.history):  # initialize with self.ind = 0
            self.history_queue.append(0)

        # Multi-step
        self.horizon = int(horizon)

        self.prioritized = bool(prioritized)
        self.priority = torch.zeros(self.max_size, device=self.device) if self.prioritized else []
        self.max_priority = float(initial_priority)

        # Sampling mask: 1 means sample-able; 0 means masked out
        self.mask = torch.zeros(self.max_size, device=self.device)

        # Actual storage
        self.obs = np.zeros((self.max_size, *self.obs_shape), dtype=self.obs_dtype)
        self.action_reward_notdone = np.zeros((self.max_size, action_dim + 2), dtype=np.float32)

        self.action_dim = int(action_dim)

    # Extract the most recent obs from the state that includes history.
    def extract_obs(self, state: Union[np.ndarray, "Any"]):
        if isinstance(state, np.ndarray):
            return state[-self.num_channels:].reshape(self.obs_shape).astype(self.obs_dtype, copy=False)
        else:
            arr = np.asarray(state)
            return arr[-self.num_channels:].reshape(self.obs_shape).astype(self.obs_dtype, copy=False)

    def one_hot_or_normalize(self, action: Union[int, float, np.ndarray]):
        if isinstance(action, (np.integer, int)):
            one_hot = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[int(action)] = 1.0
            return one_hot
        return np.asarray(action, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        self.obs[self.ind] = self.extract_obs(state)
        self.action_reward_notdone[self.ind, 0] = float(reward)
        self.action_reward_notdone[self.ind, 1] = 1.0 - float(bool(terminated))
        self.action_reward_notdone[self.ind, 2:] = self.one_hot_or_normalize(action)

        if self.prioritized:
            self.priority[self.ind] = self.max_priority

        # Tracking
        self.size = max(self.size, self.ind + 1)
        self.ep_timesteps += 1
        if terminated:
            self.env_terminates = True

        # Masking
        self.mask[(self.ind + self.history - 1) % self.max_size] = 0.0
        if self.ep_timesteps > self.horizon:  # Allow states that have a completed horizon to be sampled.
            self.mask[(self.ind - self.horizon) % self.max_size] = 1.0

        # History
        next_ind = (self.ind + 1) % self.max_size
        self.state_ind[self.ind] = np.array(self.history_queue, dtype=np.int32)
        self.history_queue.append(next_ind)
        self.next_ind[self.ind] = np.array(self.history_queue, dtype=np.int32)
        self.ind = next_ind

        # Handle episode end
        if terminated or truncated:
            self.terminal(next_state, truncated)

    def terminal(self, state: np.ndarray, truncated: bool):
        self.obs[self.ind] = self.extract_obs(state)

        # Mask out the trailing incomplete items; if truncated, also mask the past horizon steps
        self.mask[(self.ind + self.history - 1) % self.max_size] = 0.0
        past_len = min(self.ep_timesteps, self.horizon)
        past_ind = (self.ind - np.arange(past_len) - 1) % self.max_size
        self.mask[past_ind] = 0.0 if truncated else 1.0

        self.ind = (self.ind + 1) % self.max_size
        self.ep_timesteps = 0

        for _ in range(self.history):
            self.history_queue.append(self.ind)

    def sample_ind(self, prioritized=True):
        if self.prioritized & prioritized:
            try:
                csum = torch.cumsum(self.priority * self.mask, 0)
                self.sampled_ind = torch.searchsorted(
                    csum,
                    torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
                ).cpu().data.numpy()
            except Exception as e:
                print("\n=== [ReplayBuffer.sample_ind ERROR DEBUG INFO] ===")
                print(f"Exception: {repr(e)}")
                print(f"Exception: {repr(e)}")
                print(f"device: {self.device}")
                print(f"self.ind: {self.ind}, self.size: {self.size}")
                print(f"self.max_priority: {self.max_priority}")
                print(f"self.priority: {self.priority}")
                print(f"self.mask: {self.mask}")
                print(f"csum: {csum}")
        else:
            nz = torch.nonzero(self.mask).reshape(-1)
            self.sampled_ind = np.random.randint(nz.shape[0], size=self.batch_size)
            self.sampled_ind = nz[self.sampled_ind].cpu().data.numpy()
        return self.sampled_ind

    def sample(self, horizon: int, include_intermediate: bool = False, prioritized: bool = True):
        ind0 = self.sample_ind(prioritized)
        ind = (ind0.reshape(-1, 1) + np.arange(horizon).reshape(1, -1)) % self.max_size

        ard = torch.from_numpy(self.action_reward_notdone[ind]).to(self.device)

        # Sample subtrajectory (with horizon dimension) for unrolling dynamics.
        if include_intermediate:
            # Group (state, next_state) to speed up CPU -> GPU transfer.
            state_ind = np.concatenate(
                [self.state_ind[ind], self.next_ind[ind[:, -1].reshape(-1, 1)]],
                axis=1,
            )
            both_state = torch.from_numpy(self.obs[state_ind]).to(self.device).reshape(self.batch_size, -1, *self.state_shape).float()
            state = both_state[:, :-1]          # State: (batch_size, horizon, *state_dim)
            next_state = both_state[:, 1:]      # Next state: (batch_size, horizon, *state_dim)
            action = ard[:, :, 2:]              # Action: (batch_size, horizon, action_dim)
        else:
            state_ind = np.concatenate(
                [self.state_ind[ind[:, 0].reshape(-1, 1)], self.next_ind[ind[:, -1].reshape(-1, 1)]],
                axis=1,
            )
            both_state = torch.from_numpy(self.obs[state_ind]).to(self.device).reshape(self.batch_size, 2, *self.state_shape).float()
            state = both_state[:, 0]            # State: (batch_size, *state_dim)
            next_state = both_state[:, 1]       # Next state: (batch_size, *state_dim)
            action = ard[:, 0, 2:]              # Action: (batch_size, action_dim)

        rewards = ard[:, :, 0].unsqueeze(-1)
        notdone = ard[:, :, 1].unsqueeze(-1)

        return state, action, next_state, rewards, notdone

    def update_priority(self, priority: torch.Tensor):
        priority = priority.detach()
        self.priority[self.sampled_ind] = priority.reshape(-1)
        self.max_priority = max(float(priority.max()), self.max_priority)


    def reward_scale(self, eps: float = 1e-8) -> float:
        if self.size == 0:
            return 1.0
        mean_abs = np.mean(np.abs(self.action_reward_notdone[:self.size, 0]))
        return float(max(mean_abs, eps))

    def save(self, save_folder: str):
        if self.prioritized:
            np.savez_compressed(
                f"{save_folder}/buffer_data",
                obs=self.obs,
                ard=self.action_reward_notdone,
                state_ind=self.state_ind,
                next_ind=self.next_ind,
                priority=self.priority.cpu().numpy(),
                mask=self.mask.cpu().numpy(),
            )
        else:
            np.savez_compressed(
                f"{save_folder}/buffer_data",
                obs=self.obs,
                ard=self.action_reward_notdone,
                state_ind=self.state_ind,
                next_ind=self.next_ind,
                mask=self.mask.cpu().numpy(),
            )

        v = ["ind", "size", "env_terminates", "history_queue", "max_priority"]
        var_dict = {k: self.__dict__[k] for k in v}

        var_dict["history_queue"] = list(self.history_queue)

        np.save(f"{save_folder}/buffer_var.npy", var_dict, allow_pickle=True)
        print("buffer is saved!")

    def load(self, save_folder: str):
        buffer_data = np.load(f"{save_folder}/buffer_data.npz", allow_pickle=True)

        self.obs = buffer_data["obs"].astype(self.obs_dtype, copy=False)
        self.action_reward_notdone = buffer_data["ard"].astype(np.float32, copy=False)
        self.state_ind = buffer_data["state_ind"].astype(np.int32, copy=False)
        self.next_ind = buffer_data["next_ind"].astype(np.int32, copy=False)
        if self.prioritized and "priority" in buffer_data:
            self.priority = buffer_data["priority"].astype(np.float32, copy=False)
        elif self.prioritized and "priority" not in buffer_data:
            self.priority = np.zeros(self.max_size, dtype=np.float32)
        self.mask = buffer_data["mask"].astype(np.float32, copy=False)

        var_dict = np.load(f"{save_folder}/buffer_var.npy", allow_pickle=True).item()

        self.ind = int(var_dict.get("ind", 0))
        self.size = int(var_dict.get("size", 0))
        self.env_terminates = bool(var_dict.get("env_terminates", False))
        self.max_priority = float(var_dict.get("max_priority", 1.0))

        hq = var_dict.get("history_queue", [0] * self.history)
        self.history_queue = deque(hq, maxlen=self.history)
