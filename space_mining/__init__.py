"""Top-level public API for space_mining.
Expose lightweight factories and keep imports fast.
"""

from importlib import import_module
from typing import Any

# Register Gymnasium IDs on import for convenience
import_module(".envs", package=__name__)


def make_env(*args: Any, **kwargs: Any):
    """Create and return the SpaceMining environment instance.

    This function lazy-imports from `space_mining.envs` to avoid heavy imports
    at package import time.
    """
    mod = import_module(".envs", package=__name__)
    return mod.make_env(*args, **kwargs)


def register_envs(*args: Any, **kwargs: Any):
    """Register Gymnasium IDs for SpaceMining environments."""
    mod = import_module(".envs", package=__name__)
    return mod.register_envs(*args, **kwargs)


def PPOAgent(*args: Any, **kwargs: Any):
    """Factory for the PPOAgent wrapper class.

    Returns the class from `agents.ppo_agent` while avoiding importing
    Stable-Baselines3 at top-level unless needed.
    """
    mod = import_module(".agents.ppo_agent", package=__name__)
    return mod.PPOAgent(*args, **kwargs)


def save_gif(*args: Any, **kwargs: Any):
    """Save frames as a GIF via the scripts module."""
    mod = import_module(".scripts.make_gif", package=__name__)
    return mod.save_gif(*args, **kwargs)


__version__ = "0.1.2"
__all__ = ["make_env", "register_envs", "PPOAgent", "save_gif"]
