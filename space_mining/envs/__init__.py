"""Environment module for space_mining.

This module provides the core SpaceMining environment, rendering system, and utilities
for creating and managing space mining simulation environments.

The module includes:
- SpaceMining: Core Gymnasium-compatible environment for space mining simulation
- Renderer: Advanced pygame-based visualization system
- Wrappers: Environment wrappers for enhanced functionality
- make_env: Convenient factory function for creating configured environments

Examples:
    # Create a basic environment
    >>> from space_mining.envs import make_env
    >>> env = make_env()
    >>> observation, info = env.reset()
    
    # Create environment with custom parameters
    >>> env = make_env(
    ...     max_episode_steps=2000,
    ...     max_asteroids=20,
    ...     render_mode="human"
    ... )
    
    # Direct environment creation
    >>> from space_mining.envs import SpaceMining
    >>> env = SpaceMining(grid_size=100, observation_radius=20)
"""

from typing import Optional, Dict, Any, Union
import warnings

try:
    from gymnasium.envs.registration import register, registry
    import gymnasium as gym
except ImportError as e:
    raise ImportError(
        "gymnasium is required for space_mining environments. "
        "Install with: pip install gymnasium"
    ) from e

from .space_mining import SpaceMining
from .renderer import Renderer


def register_envs() -> None:
    """Register the SpaceMining environment with Gymnasium.
    
    This function registers the SpaceMining environment with Gymnasium's
    environment registry, making it available via gym.make().
    
    Raises:
        ImportError: If gymnasium is not available.
    """
    env_id = "SpaceMining"
    
    # Check if already registered to avoid duplicate registration warnings
    if env_id not in registry:
        try:
            register(
                id=env_id,
                entry_point="space_mining.envs:SpaceMining",
                max_episode_steps=1200,
                kwargs={
                    'max_episode_steps': 1200,
                    'grid_size': 80,
                    'max_asteroids': 12,
                    'max_resource_per_asteroid': 40,
                    'observation_radius': 15,
                }
            )
        except Exception as e:
            warnings.warn(f"Failed to register SpaceMining environment: {e}")


def make_env(
    max_episode_steps: int = 1200,
    grid_size: int = 80,
    max_asteroids: int = 12,
    max_resource_per_asteroid: int = 40,
    observation_radius: int = 15,
    render_mode: Optional[str] = None,
    apply_wrappers: bool = True,
    **kwargs: Any
) -> gym.Env:
    """Create and return a SpaceMining environment instance with optional wrappers.
    
    This is the recommended way to create SpaceMining environments as it applies
    sensible defaults and optional wrapper configurations.
    
    Args:
        max_episode_steps: Maximum number of steps per episode.
        grid_size: Size of the simulation grid (grid_size x grid_size).
        max_asteroids: Maximum number of asteroids in the environment.
        max_resource_per_asteroid: Maximum resources per asteroid.
        observation_radius: Agent's observation radius.
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        apply_wrappers: Whether to apply default wrappers.
        **kwargs: Additional arguments passed to the environment.
    
    Returns:
        Configured SpaceMining environment, optionally wrapped.
        
    Raises:
        ValueError: If parameters are invalid.
        ImportError: If required dependencies are missing.
        
    Examples:
        # Basic usage
        >>> env = make_env()
        
        # Custom configuration
        >>> env = make_env(
        ...     max_episode_steps=2000,
        ...     grid_size=100,
        ...     max_asteroids=20,
        ...     render_mode="human"
        ... )
        
        # Without default wrappers
        >>> env = make_env(apply_wrappers=False)
    """
    # Validate parameters
    if max_episode_steps <= 0:
        raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if max_asteroids <= 0:
        raise ValueError(f"max_asteroids must be positive, got {max_asteroids}")
    if observation_radius <= 0:
        raise ValueError(f"observation_radius must be positive, got {observation_radius}")
    if render_mode not in [None, "human", "rgb_array"]:
        raise ValueError(f"render_mode must be None, 'human', or 'rgb_array', got {render_mode}")
    
    # Create base environment
    try:
        env = SpaceMining(
            max_episode_steps=max_episode_steps,
            grid_size=grid_size,
            max_asteroids=max_asteroids,
            max_resource_per_asteroid=max_resource_per_asteroid,
            observation_radius=observation_radius,
            render_mode=render_mode,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create SpaceMining environment: {e}") from e
    
    # Apply wrappers if requested
    if apply_wrappers:
        try:
            from .wrappers import apply_default_wrappers
            env = apply_default_wrappers(env)
        except ImportError:
            # Fallback if wrappers module has issues
            warnings.warn("Could not apply default wrappers, using base environment")
        except Exception as e:
            warnings.warn(f"Failed to apply wrappers: {e}")
    
    return env


def create_env_from_config(config: Dict[str, Any]) -> gym.Env:
    """Create environment from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing environment parameters.
        
    Returns:
        Configured SpaceMining environment.
        
    Examples:
        >>> config = {
        ...     "max_episode_steps": 1500,
        ...     "grid_size": 100,
        ...     "render_mode": "human"
        ... }
        >>> env = create_env_from_config(config)
    """
    return make_env(**config)


# Register environments on module import
try:
    register_envs()
except Exception as e:
    warnings.warn(f"Environment registration failed: {e}")

__all__ = [
    "SpaceMining", 
    "Renderer",
    "make_env", 
    "register_envs",
    "create_env_from_config"
]
