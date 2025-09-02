import numpy as np

from space_mining.envs import make_env


def test_single_step_types():
    env = make_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_multiple_steps_until_done():
    env = make_env()
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


