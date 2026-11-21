import torch
from torch.distributions import Normal

from walker2d_ppo.algorithm import compute_gae, ppo_clipped_actor_loss, value_loss
from walker2d_ppo.model import GaussianActor, ValueCritic


def test_joint_log_prob_sums_action_dimension_only() -> None:
    dist = Normal(torch.zeros(3, 6), torch.ones(3, 6))
    actions = torch.randn(3, 6)
    actual = GaussianActor.joint_log_prob(dist, actions)
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, dist.log_prob(actions).sum(dim=-1))


def test_actor_and_critic_shapes_for_walker2d() -> None:
    actor = GaussianActor(obs_dim=17, action_dim=6)
    critic = ValueCritic(obs_dim=17)
    observations = torch.zeros(4, 17)
    actions, log_probs, entropy = actor.sample(observations)
    assert actions.shape == (4, 6)
    assert log_probs.shape == entropy.shape == (4,)
    assert critic(observations).shape == (4,)


def test_same_policy_same_raw_action_has_unit_ratio() -> None:
    actor = GaussianActor(17, 6)
    obs = torch.randn(8, 17)
    with torch.no_grad():
        action, old_log_prob, _ = actor.sample(obs)
    new_log_prob = actor.joint_log_prob(actor.distribution(obs), action)
    ratio = (new_log_prob - old_log_prob).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio))


def test_gae_termination_disables_bootstrap() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0]),
        values=torch.tensor([0.25]),
        next_values=torch.tensor([10.0]),
        terminated=torch.tensor([True]),
        episode_ends=torch.tensor([True]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantages, torch.tensor([0.75]))
    torch.testing.assert_close(returns, torch.tensor([1.0]))


def test_gae_truncation_bootstraps_but_stops_recursion() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 2.0]),
        values=torch.tensor([0.5, 0.25]),
        next_values=torch.tensor([4.0, 7.0]),
        terminated=torch.tensor([False, False]),
        episode_ends=torch.tensor([True, False]),  # t=0 is truncated
        gamma=0.9,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages[0], torch.tensor(4.1))
    torch.testing.assert_close(returns[0], torch.tensor(4.6))
    assert not advantages.requires_grad and not returns.requires_grad


def test_rollout_boundary_bootstraps_but_stops_after_last_sample() -> None:
    advantages, _ = compute_gae(
        rewards=torch.tensor([1.0]),
        values=torch.tensor([0.5]),
        next_values=torch.tensor([2.0]),
        terminated=torch.tensor([False]),
        episode_ends=torch.tensor([False]),
        gamma=0.9,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([2.3]))


def test_clip_behavior_for_positive_and_negative_advantages() -> None:
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.5, 1.5]))
    advantages = torch.tensor([1.0, -1.0])
    loss, ratio, clip_fraction = ppo_clipped_actor_loss(new, old, advantages, 0.2)
    torch.testing.assert_close(ratio, torch.tensor([1.5, 1.5]))
    torch.testing.assert_close(loss, torch.tensor(0.15))
    torch.testing.assert_close(clip_fraction, torch.tensor(1.0))


def test_new_actor_and_critic_predictions_receive_gradients() -> None:
    actor = GaussianActor(17, 6)
    critic = ValueCritic(17)
    obs = torch.randn(5, 17)
    action = torch.randn(5, 6)
    old_log_prob = torch.zeros(5)
    advantages = torch.ones(5)
    actor_loss, _, _ = ppo_clipped_actor_loss(
        actor.joint_log_prob(actor.distribution(obs), action), old_log_prob, advantages, 0.2
    )
    critic_loss = value_loss(critic(obs), torch.zeros(5))
    actor_loss.backward()
    critic_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert not old_log_prob.requires_grad and not advantages.requires_grad

