from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_dim: int, output_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_dim),
    )


class GaussianActor(nn.Module):
    """Factorized Gaussian over raw actions u; the environment sees clip(u)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.mean_net = _mlp(obs_dim, action_dim, hidden_size)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def joint_log_prob(dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        """Sum action coordinates only: [B, A] -> [B]."""
        return dist.log_prob(raw_action).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = self.joint_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)
        return raw_action, log_prob, entropy

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.value_net = _mlp(obs_dim, 1, hidden_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)

