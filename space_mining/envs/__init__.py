"""Environment module for space_mining.
Registers the SpaceMining environment with Gymnasium.
"""

from gymnasium.envs.registration import register, registry

from .space_mining import SpaceMining


def register_envs() -> None:
    """Register the SpaceMining environment with Gymnasium."""
    env_id = "SpaceMining"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="space_mining.envs:SpaceMining",
            max_episode_steps=1200,
        )


def make_env(**kwargs):
    """Create and return a SpaceMining environment instance, applying default wrappers."""
    from .wrappers import make_env as _wrapped_make_env

    return _wrapped_make_env(**kwargs)


# Register environments on module import
register_envs()

__all__ = ["SpaceMining", "make_env", "register_envs"]
