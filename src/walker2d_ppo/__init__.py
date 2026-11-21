"""Teaching-first PPO implementation for Gymnasium Walker2d-v5."""

from walker2d_ppo.config import PPOConfig
from walker2d_ppo.model import GaussianActor, ValueCritic

__all__ = ["GaussianActor", "PPOConfig", "ValueCritic"]

