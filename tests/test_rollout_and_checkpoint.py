from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from walker2d_ppo.config import PPOConfig
from walker2d_ppo.model import GaussianActor, ValueCritic
from walker2d_ppo.rollout import RolloutCollector
from walker2d_ppo.trainer import load_actor, save_checkpoint


class LongEpisodeEnv(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.t = 0
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.reset_count += 1
        return np.zeros(2, dtype=np.float32), {"x_position": 0.0}

    def step(self, action):
        self.t += 1
        obs = np.array([self.t, 0], dtype=np.float32)
        info = {
            "x_position": float(self.t),
            "reward_forward": 0.0,
            "reward_ctrl": 0.0,
            "reward_survive": 1.0,
        }
        return obs, 1.0, self.t >= 10, False, info


def test_episode_accumulators_survive_rollout_boundary() -> None:
    env = LongEpisodeEnv()
    actor = GaussianActor(2, 1)
    critic = ValueCritic(2)
    collector = RolloutCollector(env, actor, critic, torch.device("cpu"), seed=1)
    first = collector.collect(4, gamma=0.99, gae_lambda=0.95)
    second = collector.collect(6, gamma=0.99, gae_lambda=0.95)
    assert not first.episodes
    assert len(second.episodes) == 1
    assert second.episodes[0].episode_length == 10
    assert second.episodes[0].episode_return == 10.0
    assert env.reset_count == 2  # initial reset plus true episode end only


def test_checkpoint_roundtrip_preserves_deterministic_action(tmp_path: Path) -> None:
    actor = GaussianActor(17, 6)
    critic = ValueCritic(17)
    actor_opt = torch.optim.Adam(actor.parameters())
    critic_opt = torch.optim.Adam(critic.parameters())
    obs = torch.randn(3, 17)
    before = actor.deterministic(obs).detach()
    path = tmp_path / "checkpoint.pt"
    config = PPOConfig(output_dir=tmp_path)
    save_checkpoint(path, actor, critic, actor_opt, critic_opt, config, 123, 17, 6)
    restored, checkpoint = load_actor(path, torch.device("cpu"))
    after = restored.deterministic(obs).detach()
    torch.testing.assert_close(before, after)
    assert checkpoint["env_steps"] == 123

