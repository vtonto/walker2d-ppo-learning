import gymnasium as gym
import numpy as np


def test_walker2d_reset_step_shape_and_dtype() -> None:
    env = gym.make("Walker2d-v5")
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == (17,)
        assert obs.dtype == np.float64
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (17,)
        assert np.asarray(next_obs, dtype=np.float32).dtype == np.float32
        assert action.shape == (6,) and action.dtype == np.float32
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert {"reward_forward", "reward_ctrl", "reward_survive"} <= info.keys()
        assert env.unwrapped.dt > env.unwrapped.model.opt.timestep
        assert np.isclose(
            env.unwrapped.dt,
            env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip,
        )
    finally:
        env.close()

