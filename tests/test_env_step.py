import numpy as np

from space_mining.envs import make_env


def test_env_basic_make():
    env = make_env()
    assert env is not None
    env.close()
