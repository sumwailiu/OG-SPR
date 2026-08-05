import numpy as np
from gymnasium import spaces
import torch
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import ReplayBuffer
import copy
from typing import NamedTuple, Optional


class Samples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    next_rewards: torch.Tensor
    target_observations: torch.Tensor


# A memory-efficient prioritized replay buffer build on SB3 ReplayBuffer
class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage,
                 handle_timeout_termination, discount, add_steps, add_frame_stack):
        super(PriorityReplayBuffer, self).__init__(buffer_size=buffer_size,
                                                   observation_space=observation_space,
                                                   action_space=action_space,
                                                   device=device, n_envs=n_envs,
                                                   optimize_memory_usage=optimize_memory_usage,
                                                   handle_timeout_termination=handle_timeout_termination)
        self.additional_steps = add_steps
        self.additional_frame_stack = add_frame_stack
        self.discount = discount
        self.device = device
        self.truncate = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool)
        self.dones = self.dones.astype(np.bool)
        self.priority = torch.empty(self.buffer_size, device=self.device)
        self.max_priority = 1.0

    def update_priority(self, priority: torch.Tensor):
        self.priority[self.sampled_ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def sample_ind(self, batch_size):
        if self.full:
            mask = torch.ones_like(self.priority)
            ban_idx = []
            for i in range(self.additional_steps+1):
                ban_idx.append((self.pos-i) % self.buffer_size)
            mask[ban_idx] = 0.
        else:
            mask = torch.zeros_like(self.priority)
            mask[0 + self.additional_frame_stack: self.pos - self.additional_steps] = 1.0

        csum = torch.cumsum(self.priority * mask, 0)
        sampled_ind = torch.searchsorted(
            csum,
            torch.rand(size=(batch_size,), device=self.device) * csum[-1]
        ).cpu().data.numpy()
        return sampled_ind

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        truncation: np.ndarray,
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.truncate[self.pos] = np.array(truncation)
        self.priority[self.pos] = self.max_priority
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        batch_inds = self.sample_ind(batch_size)
        truncates = self.truncate[(batch_inds - 1) % self.buffer_size, ].flatten()
        truncates1 = self.truncate[(batch_inds - 1 + 1) % self.buffer_size,].flatten()
        truncates2 = self.truncate[(batch_inds - 1 + 2) % self.buffer_size,].flatten()
        truncations = np.logical_or(truncates, truncates1)
        truncations = np.logical_or(truncations, truncates2)

        while truncations.sum() > 0:
            batch_inds[truncations] = self.sample_ind(len(batch_inds[truncations]))

            truncates = self.truncate[(batch_inds - 1) % self.buffer_size, ].flatten()
            truncates1 = self.truncate[(batch_inds - 1 + 1) % self.buffer_size, ].flatten()
            truncates2 = self.truncate[(batch_inds - 1 + 2) % self.buffer_size, ].flatten()
            truncations = np.logical_or(truncates, truncates1)
            truncations = np.logical_or(truncations, truncates2)

        self.sampled_ind = batch_inds
        return self._get_samples(batch_inds, env=env), batch_inds

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = np.concatenate([self.observations[(batch_inds + 1 + self.additional_steps - 2) % self.buffer_size, env_indices, :],
                                       self.observations[(batch_inds + 1 + self.additional_steps - 1) % self.buffer_size, env_indices, :],
                                       self.observations[(batch_inds + 1 + self.additional_steps) % self.buffer_size, env_indices, :]], axis=1)
            target_obs = self.observations[(batch_inds + 1 + self.additional_steps - 2) % self.buffer_size, env_indices, :]
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        dones = self.dones[(batch_inds - 1) % self.buffer_size, env_indices]
        truncate = self.truncate[(batch_inds - 1) % self.buffer_size, env_indices]
        obs = self.observations[batch_inds, env_indices, :]

        for i in range(1, self.additional_frame_stack + 1):
            dones = np.logical_or(dones, self.dones[(batch_inds - 1 - i) % self.buffer_size, env_indices])
            truncate = np.logical_or(truncate, self.truncate[(batch_inds - 1 - i) % self.buffer_size, env_indices])
            select_idx = np.logical_or(dones, truncate)

            obs = np.concatenate([self.observations[(batch_inds - i) % self.buffer_size, env_indices, :], obs], axis=1)
            obs[select_idx, 0:3] = obs[select_idx, 3:6]

        rewards = self.rewards[batch_inds, env_indices]
        next_rewards = self.rewards[batch_inds, env_indices]
        not_dones = copy.deepcopy(1-self.dones[batch_inds, env_indices].astype(np.float32))
        discount = 1.0
        for i in range(self.additional_steps):
            discount *= self.discount
            not_dones *= (1 - self.dones[(batch_inds + 1 + i) % self.buffer_size, env_indices].astype(np.float32))
            rewards += self.rewards[(batch_inds + 1 + i) % self.buffer_size, env_indices] * not_dones * discount

        data = (
            obs,
            self.actions[batch_inds, env_indices, :],
            next_obs,
            ((1-not_dones) * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(rewards.reshape(-1, 1), env),
            self._normalize_reward(next_rewards.reshape(-1, 1), env),
            target_obs
        )
        return Samples(*tuple(map(self.to_torch, data)))
