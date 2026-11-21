from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from walker2d_ppo.config import profile
from walker2d_ppo.evaluate import evaluate_policy
from walker2d_ppo.trainer import load_actor, train


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train from-scratch PPO on Walker2d-v5")
    parser.add_argument("--profile", choices=["smoke", "short", "full"], default="short")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--forward-reward-weight", type=float, default=1.0)
    parser.add_argument("--ctrl-cost-weight", type=float, default=1e-3)
    parser.add_argument("--healthy-reward", type=float, default=1.0)
    return parser


def train_main() -> None:
    args = _train_parser().parse_args()
    overrides = {
        "seed": args.seed,
        "device": args.device,
        "forward_reward_weight": args.forward_reward_weight,
        "ctrl_cost_weight": args.ctrl_cost_weight,
        "healthy_reward": args.healthy_reward,
    }
    for name in ("total_timesteps", "rollout_steps", "update_epochs", "minibatch_size", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    config = profile(args.profile, **overrides)
    checkpoint = train(config)
    print(f"checkpoint: {checkpoint}")


def _evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministically evaluate a Walker2d PPO checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--video", type=Path)
    return parser


def evaluate_main() -> None:
    args = _evaluate_parser().parse_args()
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, device)
    config = checkpoint["config"]
    env_kwargs = {
        "forward_reward_weight": config["forward_reward_weight"],
        "ctrl_cost_weight": config["ctrl_cost_weight"],
        "healthy_reward": config["healthy_reward"],
    }
    result = evaluate_policy(
        actor,
        config["env_id"],
        seeds=[args.seed + i for i in range(args.episodes)],
        device=device,
        env_kwargs=env_kwargs,
        video_path=args.video,
    )
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))

