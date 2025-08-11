import numpy as np

from space_mining.envs import make_env


def test_action_space_shape_and_bounds():
    env = make_env()
    assert env.action_space.shape == (3,)
    action = env.action_space.sample()
    assert action.shape == (3,)
    assert np.all(action >= env.action_space.low)
    assert np.all(action <= env.action_space.high)
    env.close()


def test_observation_space_shape_and_contains():
    env = make_env()
    assert env.observation_space.shape == (53,)
    obs, _ = env.reset()
    assert obs.shape == (53,)
    assert env.observation_space.contains(obs)
    env.close()