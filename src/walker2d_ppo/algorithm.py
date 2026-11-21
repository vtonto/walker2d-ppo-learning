from __future__ import annotations

import torch


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    episode_ends: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed GAE targets without crossing resets or the rollout boundary.

    ``terminated`` controls TD bootstrap. ``episode_ends`` is terminated OR
    truncated and controls temporal recursion. The final buffer element also
    stops recursion because no later sample belongs to this rollout.
    """
    tensors = (rewards, values, next_values, terminated, episode_ends)
    if any(x.ndim != 1 for x in tensors):
        raise ValueError("all GAE inputs must be rank-1 tensors")
    if len({x.shape[0] for x in tensors}) != 1:
        raise ValueError("all GAE inputs must have equal length")

    bootstrap_mask = (~terminated.bool()).to(values.dtype)
    deltas = rewards + gamma * bootstrap_mask * next_values - values
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in range(rewards.shape[0] - 1, -1, -1):
        continue_mask = (~episode_ends[t].bool()).to(values.dtype)
        if t == rewards.shape[0] - 1:
            continue_mask = torch.zeros_like(continue_mask)
        gae = deltas[t] + gamma * gae_lambda * continue_mask * gae
        advantages[t] = gae
    returns = values + advantages
    return advantages.detach(), returns.detach()


def ppo_clipped_actor_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return loss, ratio, clip_fraction


def value_loss(predicted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * (predicted - targets).pow(2).mean()

