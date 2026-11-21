from __future__ import annotations

from dataclasses import dataclass

import torch

from walker2d_ppo.algorithm import ppo_clipped_actor_loss, value_loss
from walker2d_ppo.config import PPOConfig
from walker2d_ppo.model import GaussianActor, ValueCritic
from walker2d_ppo.rollout import RolloutBatch


@dataclass(slots=True)
class UpdateMetrics:
    actor_loss: float
    critic_loss: float
    approx_kl: float
    clip_fraction: float
    entropy: float
    action_mean: float
    action_std: float
    action_saturation: float


def ppo_update(
    actor: GaussianActor,
    critic: ValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
) -> UpdateMetrics:
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = advantages.shape[0]
    metrics: list[tuple[float, ...]] = []

    for _ in range(config.update_epochs):
        for indices in torch.randperm(n, device=advantages.device).split(config.minibatch_size):
            obs = batch.observations[indices]
            raw_actions = batch.raw_actions[indices]
            old_log_probs = batch.old_log_probs[indices]
            mb_advantages = advantages[indices]
            targets = batch.returns[indices]

            dist = actor.distribution(obs)
            new_log_probs = actor.joint_log_prob(dist, raw_actions)
            actor_loss, ratio, clip_fraction = ppo_clipped_actor_loss(
                new_log_probs, old_log_probs, mb_advantages, config.clip_coef
            )
            entropy = dist.entropy().sum(dim=-1).mean()
            total_actor_loss = actor_loss - config.entropy_coef * entropy
            actor_optimizer.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_optimizer.step()

            predicted_values = critic(obs)
            critic_loss = value_loss(predicted_values, targets)
            critic_optimizer.zero_grad(set_to_none=True)
            (config.value_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            metrics.append(
                (
                    float(actor_loss.detach()), float(critic_loss.detach()),
                    float(approx_kl), float(clip_fraction), float(entropy.detach()),
                )
            )

    means = torch.tensor(metrics).mean(dim=0).tolist()
    with torch.no_grad():
        dist = actor.distribution(batch.observations)
        action_mean = float(dist.mean.mean())
        action_std = float(dist.stddev.mean())
        saturation = float((batch.env_actions.abs() >= 0.999).float().mean())
    return UpdateMetrics(*means, action_mean, action_std, saturation)

