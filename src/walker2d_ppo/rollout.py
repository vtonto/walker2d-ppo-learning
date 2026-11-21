from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from walker2d_ppo.algorithm import compute_gae
from walker2d_ppo.model import GaussianActor, ValueCritic


@dataclass(slots=True)
class EpisodeRecord:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    forward_distance: float


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    next_values: torch.Tensor
    terminated: torch.Tensor
    episode_ends: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    reward_forward: torch.Tensor
    reward_ctrl: torch.Tensor
    reward_survive: torch.Tensor
    env_actions: torch.Tensor
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def assert_valid(self, obs_dim: int, action_dim: int) -> None:
        n = self.rewards.shape[0]
        assert self.observations.shape == (n, obs_dim)
        assert self.raw_actions.shape == (n, action_dim)
        assert self.env_actions.shape == (n, action_dim)
        for name in (
            "old_log_probs", "old_values", "next_values", "terminated",
            "episode_ends", "advantages", "returns", "reward_forward",
            "reward_ctrl", "reward_survive",
        ):
            assert getattr(self, name).shape == (n,), name
        for tensor in (
            self.observations, self.raw_actions, self.old_log_probs,
            self.old_values, self.rewards, self.next_values,
            self.advantages, self.returns, self.reward_forward,
            self.reward_ctrl, self.reward_survive, self.env_actions,
        ):
            assert torch.isfinite(tensor).all()
            assert not tensor.requires_grad


class RolloutCollector:
    """Stateful collector: rollout boundaries do not reset an episode."""

    def __init__(
        self,
        env: gym.Env,
        actor: GaussianActor,
        critic: ValueCritic,
        device: torch.device,
        seed: int,
    ) -> None:
        self.env = env
        self.actor = actor
        self.critic = critic
        self.device = device
        obs, info = env.reset(seed=seed)
        self.obs = np.asarray(obs, dtype=np.float32)
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_start_x = float(info.get("x_position", 0.0))
        self.last_x = self.episode_start_x

    def collect(
        self,
        steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        data: dict[str, list[Any]] = {
            key: []
            for key in (
                "observations", "raw_actions", "old_log_probs", "old_values",
                "rewards", "next_values", "terminated", "episode_ends",
                "reward_forward", "reward_ctrl", "reward_survive", "env_actions",
            )
        }
        episodes: list[EpisodeRecord] = []
        action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float32)

        for _ in range(steps):
            obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                raw_action_tensor, log_prob_tensor, _ = self.actor.sample(obs_tensor)
                value_tensor = self.critic(obs_tensor)
            raw_action = raw_action_tensor.squeeze(0).cpu().numpy()
            env_action = np.clip(raw_action, action_low, action_high).astype(np.float32)
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)
            next_obs_f32 = np.asarray(next_obs, dtype=np.float32)
            with torch.no_grad():
                next_value = self.critic(
                    torch.as_tensor(next_obs_f32, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).squeeze(0)

            data["observations"].append(obs_tensor.squeeze(0))
            data["raw_actions"].append(raw_action_tensor.squeeze(0))
            data["old_log_probs"].append(log_prob_tensor.squeeze(0))
            data["old_values"].append(value_tensor.squeeze(0))
            data["rewards"].append(float(reward))
            data["next_values"].append(next_value)
            data["terminated"].append(bool(terminated))
            data["episode_ends"].append(bool(terminated or truncated))
            data["reward_forward"].append(float(info.get("reward_forward", np.nan)))
            data["reward_ctrl"].append(float(info.get("reward_ctrl", np.nan)))
            data["reward_survive"].append(float(info.get("reward_survive", np.nan)))
            data["env_actions"].append(torch.as_tensor(env_action, device=self.device))

            self.episode_return += float(reward)
            self.episode_length += 1
            self.last_x = float(info.get("x_position", self.last_x))
            self.obs = next_obs_f32

            if terminated or truncated:
                episodes.append(
                    EpisodeRecord(
                        episode_return=self.episode_return,
                        episode_length=self.episode_length,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        forward_distance=self.last_x - self.episode_start_x,
                    )
                )
                reset_obs, reset_info = self.env.reset()
                self.obs = np.asarray(reset_obs, dtype=np.float32)
                self.episode_return = 0.0
                self.episode_length = 0
                self.episode_start_x = float(reset_info.get("x_position", 0.0))
                self.last_x = self.episode_start_x

        def stack(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            items = data[name]
            if isinstance(items[0], torch.Tensor):
                result = torch.stack(items).detach()
            else:
                result = torch.as_tensor(items, device=self.device)
            return result.to(dtype=dtype) if dtype is not None else result

        rewards = stack("rewards", torch.float32)
        values = stack("old_values", torch.float32)
        next_values = stack("next_values", torch.float32)
        terminated_tensor = stack("terminated", torch.bool)
        episode_ends = stack("episode_ends", torch.bool)
        advantages, returns = compute_gae(
            rewards, values, next_values, terminated_tensor, episode_ends, gamma, gae_lambda
        )
        batch = RolloutBatch(
            observations=stack("observations", torch.float32),
            raw_actions=stack("raw_actions", torch.float32),
            old_log_probs=stack("old_log_probs", torch.float32),
            old_values=values,
            rewards=rewards,
            next_values=next_values,
            terminated=terminated_tensor,
            episode_ends=episode_ends,
            advantages=advantages,
            returns=returns,
            reward_forward=stack("reward_forward", torch.float32),
            reward_ctrl=stack("reward_ctrl", torch.float32),
            reward_survive=stack("reward_survive", torch.float32),
            env_actions=stack("env_actions", torch.float32),
            episodes=episodes,
        )
        return batch

