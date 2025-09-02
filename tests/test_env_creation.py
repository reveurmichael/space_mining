import numpy as np

from space_mining.envs import make_env


def test_env_creation_and_reset():
    env = make_env()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    env.close()