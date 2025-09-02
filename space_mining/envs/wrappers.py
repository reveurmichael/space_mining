from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .space_mining import SpaceMining


class FlattenActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Propagate commonly tweaked attributes set on the wrapper to the underlying env
        for attr_name in (
            "agent_energy",
            "agent_inventory",
            "agent_position",
            "agent_velocity",
        ):
            if attr_name in self.__dict__:
                setattr(self.env, attr_name, getattr(self, attr_name))
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()


def make_env(**kwargs) -> gym.Env:
    """
    Create and return the SpaceMining environment with default wrappers.

    Returns:
        A wrapped environment compatible with standard RL algorithms
    """
    env = SpaceMining(**kwargs)
    env = FlattenActionSpaceWrapper(env)
    return env
