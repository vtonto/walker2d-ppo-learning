from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from walker2d_ppo.config import PPOConfig
from walker2d_ppo.evaluate import evaluate_policy
from walker2d_ppo.model import GaussianActor, ValueCritic
from walker2d_ppo.rollout import RolloutCollector
from walker2d_ppo.update import ppo_update


METRIC_FIELDS = [
    "env_steps", "update", "steps_per_second", "update_seconds",
    "actor_loss", "critic_loss", "approx_kl", "clip_fraction", "entropy",
    "action_mean", "action_std", "action_saturation", "mean_rollout_reward",
    "mean_reward_forward", "mean_reward_ctrl", "mean_reward_survive",
    "completed_episodes", "mean_episode_return", "mean_episode_length",
    "termination_fraction", "truncation_fraction",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def save_checkpoint(
    path: Path,
    actor: GaussianActor,
    critic: ValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    env_steps: int,
    obs_dim: int,
    action_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "config": config.as_serializable_dict(),
            "env_steps": env_steps,
            "seed": config.seed,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "note": "Model/optimizer RNG is saved; exact mid-episode resume is not implemented.",
        },
        path,
    )


def load_actor(path: Path, device: torch.device) -> tuple[GaussianActor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    actor = GaussianActor(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        int(config["hidden_size"]),
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, checkpoint


def train(config: PPOConfig) -> Path:
    config.validate()
    seed_everything(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.as_serializable_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    env = gym.make(config.env_id, **config.env_kwargs())
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    actor = GaussianActor(obs_dim, action_dim, config.hidden_size).to(device)
    critic = ValueCritic(obs_dim, config.hidden_size).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    collector = RolloutCollector(env, actor, critic, device, config.seed)

    metrics_path = output_dir / "metrics.csv"
    env_steps = 0
    update_index = 0
    start_time = time.perf_counter()
    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            while env_steps < config.total_timesteps:
                steps = min(config.rollout_steps, config.total_timesteps - env_steps)
                batch = collector.collect(steps, config.gamma, config.gae_lambda)
                batch.assert_valid(obs_dim, action_dim)
                env_steps += steps
                update_started = time.perf_counter()
                update_metrics = ppo_update(
                    actor, critic, actor_optimizer, critic_optimizer, batch, config
                )
                update_seconds = time.perf_counter() - update_started
                update_index += 1
                elapsed = time.perf_counter() - start_time
                episode_returns = [ep.episode_return for ep in batch.episodes]
                episode_lengths = [float(ep.episode_length) for ep in batch.episodes]
                completed = len(batch.episodes)
                row = {
                    "env_steps": env_steps,
                    "update": update_index,
                    "steps_per_second": env_steps / max(elapsed, 1e-9),
                    "update_seconds": update_seconds,
                    **asdict(update_metrics),
                    "mean_rollout_reward": float(batch.rewards.mean()),
                    "mean_reward_forward": float(batch.reward_forward.mean()),
                    "mean_reward_ctrl": float(batch.reward_ctrl.mean()),
                    "mean_reward_survive": float(batch.reward_survive.mean()),
                    "completed_episodes": completed,
                    "mean_episode_return": _mean(episode_returns),
                    "mean_episode_length": _mean(episode_lengths),
                    "termination_fraction": (
                        sum(ep.terminated for ep in batch.episodes) / completed if completed else float("nan")
                    ),
                    "truncation_fraction": (
                        sum(ep.truncated for ep in batch.episodes) / completed if completed else float("nan")
                    ),
                }
                writer.writerow(row)
                handle.flush()
    finally:
        env.close()

    checkpoint_path = output_dir / "final.pt"
    save_checkpoint(
        checkpoint_path, actor, critic, actor_optimizer, critic_optimizer,
        config, env_steps, obs_dim, action_dim,
    )
    eval_result = evaluate_policy(
        actor,
        config.env_id,
        seeds=[10_000 + config.seed + i for i in range(config.eval_episodes)],
        device=device,
        env_kwargs=config.env_kwargs(),
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(eval_result.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return checkpoint_path

