from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import cv2
from collections import deque


class DMC2GymWrapper(gym.Env):
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        action_repeat: int = 2,
        flat_observation: bool = True,
        *,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        from dm_control import suite
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
            environment_kwargs={"flat_observation": flat_observation}
        )
        self.action_repeat = action_repeat
        self.render_mode = render_mode
        self._flatten_observation = None

        self._obs_key_order: list[str] = []
        self._obs_splits: np.ndarray
        self.observation_space = self._convert_observation_spec(
            self._env.observation_spec()
        )
        self.action_space = self._convert_action_spec(self._env.action_spec())
        self.max_episode_length = 1000

    def reset(self, *, seed: int | None = None, options=None):
        self.t = 0  # Reset internal timer
        if seed is not None:
            self._env.task.random.set_seed(int(seed))
        timestep = self._env.reset()

        return torch.tensor(
            np.concatenate(
                [np.asarray([obs]) if obs.size == 1 else obs for obs in timestep.observation.values()], axis=0
            ),
            dtype=torch.float32,
        ).unsqueeze(dim=0)

    def step(self, action):
        reward = 0
        for k in range(self.action_repeat):
            timestep = self._env.step(action)
            r = timestep.reward if timestep.reward is not None else 0
            reward += r
            self.t += 1  # Increment internal timer
            done = timestep.last() or (self.t == self.max_episode_length)
            if done:
                break
        obs = torch.tensor(
            np.concatenate(
                [np.asarray([obs]) if obs.size == 1 else obs for obs in timestep.observation.values()], axis=0
            ),
            dtype=torch.float32,
        ).unsqueeze(dim=0)

        terminated = timestep.last() & (timestep.discount == 0)
        truncated = self.t == self.max_episode_length
        return obs, reward, terminated, truncated

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("undefined render_mode")
        frame = self._env.physics.render(height=480, width=640, camera_id=0)

        if self.render_mode == "rgb_array":
            return frame
        elif self.render_mode == "human":
            import matplotlib.pyplot as plt

            plt.imshow(frame)
            plt.axis("off")
            plt.show(block=False)
        else:
            raise NotImplementedError(f"unknown render_mode: {self.render_mode}")

    def close(self):
        self._env.close()

    @staticmethod
    def _convert_action_spec(spec):
        low = spec.minimum.astype(np.float32)
        high = spec.maximum.astype(np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _convert_observation_spec(self, obs_spec):
        self._obs_key_order.clear()
        elem_counts = []

        for key, val in obs_spec.items():
            self._obs_key_order.append(key)
            elem_counts.append(int(np.prod(val.shape, dtype=int)))

        total_dim = int(sum(elem_counts))
        self._obs_splits = np.add.accumulate([0] + elem_counts)

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _obs_from_timestep(self, timestep):
        obs = timestep.observation
        return (
            obs[next(iter(obs))]
            if self._flatten_observation
            else {k: np.asarray(v) for k, v in obs.items()}
        )


class DMCVisual2GymWrapper(gym.Env):
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        symbolic,
        bit_depth,
        image_size,
        *,
        action_repeat: int = 2,
        seed: int = 0
    ):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
        )
        self.action_repeat = action_repeat
        self._flatten_observation = None

        self._obs_key_order: list[str] = []
        self._obs_splits: np.ndarray
        self.observation_space = self._convert_observation_spec(
            self._env.observation_spec()
        )
        self.action_space = self._convert_action_spec(self._env.action_spec())
        self.max_episode_length = 1000
        self.symbolic = symbolic
        self.image_size = image_size
        self.bit_depth = bit_depth
        self.camera_id = 2 if domain_name == 'quadruped' else 0
        if not symbolic:
            self._env = pixels.Wrapper(self._env)
            self.observation_space = spaces.Box(low=0,
                                                high=255,
                                                shape=(3, *self.image_size),
                                                dtype=np.uint8)

    def reset(self, *, seed: int = 0, options=None):
        # Initialize the RNG if the seed is manually passed
        self.t = 0  # Reset internal timer
        timestep = self._env.reset()
        if self.symbolic:
            return torch.tensor(
                np.concatenate(
                    [np.asarray([obs]) if obs.size == 1 else obs for obs in timestep.observation.values()], axis=0
                ),
                dtype=torch.float32,
            ).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.physics.render(*self.image_size, camera_id=self.camera_id), self.bit_depth)

    def step(self, action):
        reward = 0
        for k in range(self.action_repeat):
            timestep = self._env.step(action)
            r = timestep.reward if timestep.reward is not None else 0
            reward += r
            self.t += 1  # Increment internal timer
            done = timestep.last() or (self.t == self.max_episode_length)
            if done:
                break
        if self.symbolic:
            obs = torch.tensor(
                np.concatenate(
                    [np.asarray([obs]) if obs.size == 1 else obs for obs in timestep.observation.values()], axis=0
                ),
                dtype=torch.float32,
            ).unsqueeze(dim=0)
        else:
            obs = _images_to_observation(self._env.physics.render(*self.image_size, camera_id=self.camera_id), self.bit_depth)

        terminated = timestep.last() & (timestep.discount == 0)
        truncated = (self.t == self.max_episode_length) & (timestep.discount != 0)
        return obs, reward, terminated, truncated

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=self.camera_id)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @staticmethod
    def _convert_action_spec(spec):
        # dm_control action spec to gym Box
        low = spec.minimum.astype(np.float32)
        high = spec.maximum.astype(np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _convert_observation_spec(self, obs_spec):
        self._obs_key_order.clear()
        elem_counts = []

        for key, val in obs_spec.items():
            self._obs_key_order.append(key)
            elem_counts.append(int(np.prod(val.shape, dtype=int)))

        total_dim = int(sum(elem_counts))
        self._obs_splits = np.add.accumulate([0] + elem_counts)

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _obs_from_timestep(self, timestep):
        obs = timestep.observation
        return (
            obs[next(iter(obs))]
            if self._flatten_observation
            else {k: np.asarray(v) for k, v in obs.items()}
        )


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: deque[torch.Tensor] = deque(maxlen=num_stack)

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(3*num_stack, *self.env.image_size),
                                            dtype=np.uint8)

    def reset(self, *, seed: int = 0, options=None):
        obs = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs.clone())
        return self._get_stacked()

    def step(self, action):
        obs, reward, terminated, truncated = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked(), reward, terminated, truncated

    def _get_stacked(self) -> torch.Tensor:
        return torch.cat(list(self.frames), dim=1)


def _images_to_observation(images, bit_depth):
    transposed_images = images.transpose(2, 0, 1).copy()
    images = torch.tensor(transposed_images, dtype=torch.uint8)  # Resize and put channel first
    return images.unsqueeze(dim=0)  # Add batch dimension
