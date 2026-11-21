from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from walker2d_ppo.model import GaussianActor


@dataclass(slots=True)
class EvaluationResult:
    returns: list[float]
    lengths: list[int]
    distances: list[float]
    control_energy: list[float]

    def summary(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, values in asdict(self).items():
            array = np.asarray(values, dtype=np.float64)
            result[f"mean_{name}"] = float(array.mean())
            result[f"std_{name}"] = float(array.std())
        return result


@torch.no_grad()
def evaluate_policy(
    actor: GaussianActor,
    env_id: str,
    seeds: list[int],
    device: torch.device,
    env_kwargs: dict[str, float] | None = None,
    video_path: Path | None = None,
) -> EvaluationResult:
    render_mode = "rgb_array" if video_path is not None else None
    env = gym.make(env_id, render_mode=render_mode, **(env_kwargs or {}))
    returns: list[float] = []
    lengths: list[int] = []
    distances: list[float] = []
    energies: list[float] = []
    frames: list[np.ndarray] = []
    try:
        for episode_index, seed in enumerate(seeds):
            obs, info = env.reset(seed=seed)
            start_x = float(info.get("x_position", 0.0))
            last_x = start_x
            episode_return = 0.0
            episode_length = 0
            energy = 0.0
            while True:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                raw_action = actor.deterministic(obs_tensor).squeeze(0).cpu().numpy()
                action = np.clip(raw_action, env.action_space.low, env.action_space.high).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                energy += float(np.square(action).sum())
                last_x = float(info.get("x_position", last_x))
                if video_path is not None and episode_index == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            distances.append(last_x - start_x)
            energies.append(energy)
    finally:
        env.close()
    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30, loop=0)
    return EvaluationResult(returns, lengths, distances, energies)

