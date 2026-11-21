from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(slots=True)
class PPOConfig:
    """All tunable settings in one inspectable place."""

    env_id: str = "Walker2d-v5"
    total_timesteps: int = 200_000
    rollout_steps: int = 2_048
    update_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    seed: int = 1
    device: str = "cpu"
    eval_episodes: int = 3
    output_dir: Path = Path("runs/baseline")
    # Passed directly to Walker2d-v5. Keep defaults explicit for experiments.
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0

    def validate(self) -> None:
        if self.total_timesteps <= 0 or self.rollout_steps <= 0:
            raise ValueError("total_timesteps and rollout_steps must be positive")
        if not 0 < self.minibatch_size <= self.rollout_steps:
            raise ValueError("minibatch_size must be in [1, rollout_steps]")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.clip_coef <= 0:
            raise ValueError("clip_coef must be positive")
        if self.device not in {"cpu", "mps"}:
            raise ValueError("device must be 'cpu' or 'mps'")

    def env_kwargs(self) -> dict[str, float]:
        return {
            "forward_reward_weight": self.forward_reward_weight,
            "ctrl_cost_weight": self.ctrl_cost_weight,
            "healthy_reward": self.healthy_reward,
        }

    def as_serializable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result


def profile(name: str, **overrides: object) -> PPOConfig:
    """Return a documented budget profile; convergence is never implied."""
    profiles = {
        "smoke": PPOConfig(
            total_timesteps=128,
            rollout_steps=128,
            update_epochs=1,
            minibatch_size=64,
            eval_episodes=1,
            output_dir=Path("runs/smoke"),
        ),
        "short": PPOConfig(
            total_timesteps=20_000,
            rollout_steps=1_024,
            update_epochs=5,
            minibatch_size=64,
            eval_episodes=3,
            output_dir=Path("runs/short"),
        ),
        "full": PPOConfig(),
    }
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(profiles)}")
    return replace(profiles[name], **overrides)

