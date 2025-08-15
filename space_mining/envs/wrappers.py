"""Environment wrappers for space_mining.

This module provides various wrappers to enhance and modify the behavior of
the SpaceMining environment. Wrappers follow the Gymnasium wrapper pattern
and can be composed to create complex environment configurations.

Available wrappers:
- FlattenActionSpaceWrapper: Handles action space flattening and state propagation
- ObservationNormalizationWrapper: Normalizes observations for better training
- RewardScalingWrapper: Scales and clips rewards for stable training
- FrameStackWrapper: Stacks multiple observations for temporal context
- RecordingWrapper: Records episodes for analysis and replay

Examples:
    # Apply single wrapper
    >>> from space_mining.envs import SpaceMining
    >>> from space_mining.envs.wrappers import ObservationNormalizationWrapper
    >>> env = SpaceMining()
    >>> env = ObservationNormalizationWrapper(env)
    
    # Apply multiple wrappers
    >>> env = apply_default_wrappers(env)
    
    # Custom wrapper configuration
    >>> env = RewardScalingWrapper(env, scale=0.1, clip_range=(-10, 10))
"""

from typing import Any, Dict, Tuple, Optional, List, Union
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .space_mining import SpaceMining


class FlattenActionSpaceWrapper(gym.Wrapper):
    """Wrapper that handles action space flattening and state propagation.
    
    This wrapper ensures that commonly modified attributes on the wrapper
    are properly propagated to the underlying environment during step execution.
    This is particularly useful for environments that need to maintain state
    synchronization between wrapper and base environment.
    
    Attributes propagated:
    - agent_energy: Current agent energy level
    - agent_inventory: Current agent inventory amount
    - agent_position: Current agent position
    - agent_velocity: Current agent velocity
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the wrapper.
        
        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._propagated_attrs = [
            "agent_energy",
            "agent_inventory", 
            "agent_position",
            "agent_velocity",
        ]

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            **kwargs: Arguments passed to underlying environment reset.
            
        Returns:
            Tuple of (observation, info) from the reset environment.
        """
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Propagates wrapper attributes to the underlying environment before
        executing the step to maintain state synchronization.
        
        Args:
            action: Action to execute.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Propagate commonly tweaked attributes from wrapper to underlying env
        for attr_name in self._propagated_attrs:
            if hasattr(self, attr_name):
                setattr(self.env, attr_name, getattr(self, attr_name))
        
        return self.env.step(action)

    def render(self, **kwargs) -> Optional[np.ndarray]:
        """Render the environment.
        
        Args:
            **kwargs: Arguments passed to underlying environment render.
            
        Returns:
            Rendered frame if render_mode is 'rgb_array', None otherwise.
        """
        return self.env.render(**kwargs)

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


class ObservationNormalizationWrapper(gym.ObservationWrapper):
    """Wrapper that normalizes observations for stable training.
    
    This wrapper maintains running statistics of observations and normalizes
    them to have zero mean and unit variance. This can significantly improve
    training stability for some algorithms.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8) -> None:
        """Initialize the normalization wrapper.
        
        Args:
            env: Environment to wrap.
            epsilon: Small constant to avoid division by zero.
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float64)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float64)
        self.count = 0

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize the observation.
        
        Args:
            observation: Raw observation from environment.
            
        Returns:
            Normalized observation.
        """
        # Update running statistics
        self.count += 1
        delta = observation - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = observation - self.obs_mean
        self.obs_var += delta * delta2
        
        # Normalize observation
        if self.count > 1:
            var = self.obs_var / (self.count - 1)
            return (observation - self.obs_mean) / np.sqrt(var + self.epsilon)
        else:
            return observation - self.obs_mean


class RewardScalingWrapper(gym.RewardWrapper):
    """Wrapper that scales and clips rewards for stable training.
    
    This wrapper applies scaling and optional clipping to rewards to
    prevent extremely large or small rewards that can destabilize training.
    """

    def __init__(
        self, 
        env: gym.Env, 
        scale: float = 1.0, 
        clip_range: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize the reward scaling wrapper.
        
        Args:
            env: Environment to wrap.
            scale: Scaling factor for rewards.
            clip_range: Optional tuple of (min, max) for reward clipping.
        """
        super().__init__(env)
        self.scale = scale
        self.clip_range = clip_range

    def reward(self, reward: float) -> float:
        """Scale and optionally clip the reward.
        
        Args:
            reward: Raw reward from environment.
            
        Returns:
            Scaled and optionally clipped reward.
        """
        scaled_reward = reward * self.scale
        
        if self.clip_range is not None:
            scaled_reward = np.clip(scaled_reward, self.clip_range[0], self.clip_range[1])
            
        return scaled_reward


class FrameStackWrapper(gym.ObservationWrapper):
    """Wrapper that stacks multiple observations for temporal context.
    
    This wrapper maintains a buffer of the last N observations and returns
    them stacked together. This can help agents learn temporal patterns.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        """Initialize the frame stacking wrapper.
        
        Args:
            env: Environment to wrap.
            num_stack: Number of observations to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = []
        
        # Update observation space
        low = np.repeat(env.observation_space.low, num_stack)
        high = np.repeat(env.observation_space.high, num_stack)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and frame stack.
        
        Args:
            **kwargs: Arguments passed to underlying environment reset.
            
        Returns:
            Tuple of (stacked_observation, info).
        """
        observation, info = self.env.reset(**kwargs)
        self.frames = [observation] * self.num_stack
        return self.observation(observation), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Stack observations.
        
        Args:
            observation: Latest observation.
            
        Returns:
            Stacked observations.
        """
        self.frames.append(observation)
        if len(self.frames) > self.num_stack:
            self.frames.pop(0)
        return np.concatenate(self.frames)


class RecordingWrapper(gym.Wrapper):
    """Wrapper that records episodes for analysis and replay.
    
    This wrapper maintains a record of all observations, actions, rewards,
    and info dictionaries for the current episode. Useful for debugging
    and analysis.
    """

    def __init__(self, env: gym.Env, record_info: bool = True) -> None:
        """Initialize the recording wrapper.
        
        Args:
            env: Environment to wrap.
            record_info: Whether to record info dictionaries.
        """
        super().__init__(env)
        self.record_info = record_info
        self.reset_recording()

    def reset_recording(self) -> None:
        """Reset the episode recording."""
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminated': [],
            'truncated': [],
        }
        if self.record_info:
            self.episode_data['infos'] = []

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and recording.
        
        Args:
            **kwargs: Arguments passed to underlying environment reset.
            
        Returns:
            Tuple of (observation, info).
        """
        observation, info = self.env.reset(**kwargs)
        self.reset_recording()
        self.episode_data['observations'].append(observation.copy())
        if self.record_info:
            self.episode_data['infos'].append(info.copy())
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step and record data.
        
        Args:
            action: Action to execute.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Record step data
        self.episode_data['observations'].append(observation.copy())
        self.episode_data['actions'].append(action.copy())
        self.episode_data['rewards'].append(reward)
        self.episode_data['terminated'].append(terminated)
        self.episode_data['truncated'].append(truncated)
        if self.record_info:
            self.episode_data['infos'].append(info.copy())
        
        return observation, reward, terminated, truncated, info

    def get_episode_data(self) -> Dict[str, List[Any]]:
        """Get recorded episode data.
        
        Returns:
            Dictionary containing recorded episode data.
        """
        return self.episode_data.copy()


def apply_default_wrappers(env: gym.Env, **wrapper_kwargs) -> gym.Env:
    """Apply default wrappers to an environment.
    
    This function applies a standard set of wrappers that are commonly
    useful for training agents on the SpaceMining environment.
    
    Args:
        env: Base environment to wrap.
        **wrapper_kwargs: Additional arguments for specific wrappers.
            Supported keys:
            - normalize_obs: bool, whether to apply observation normalization
            - reward_scale: float, scale factor for rewards
            - reward_clip: tuple, min/max values for reward clipping
            - record_episodes: bool, whether to record episode data
    
    Returns:
        Wrapped environment.
        
    Examples:
        # Default wrappers
        >>> env = apply_default_wrappers(env)
        
        # Custom configuration
        >>> env = apply_default_wrappers(
        ...     env,
        ...     normalize_obs=True,
        ...     reward_scale=0.1,
        ...     reward_clip=(-10, 10)
        ... )
    """
    # Always apply the flatten action space wrapper first
    env = FlattenActionSpaceWrapper(env)
    
    # Optional observation normalization
    if wrapper_kwargs.get('normalize_obs', False):
        env = ObservationNormalizationWrapper(env)
    
    # Optional reward scaling/clipping
    reward_scale = wrapper_kwargs.get('reward_scale')
    reward_clip = wrapper_kwargs.get('reward_clip')
    if reward_scale is not None or reward_clip is not None:
        env = RewardScalingWrapper(
            env, 
            scale=reward_scale or 1.0,
            clip_range=reward_clip
        )
    
    # Optional episode recording
    if wrapper_kwargs.get('record_episodes', False):
        env = RecordingWrapper(env)
    
    return env


def make_env(**kwargs) -> gym.Env:
    """Create SpaceMining environment with default wrappers (legacy function).
    
    This function is maintained for backward compatibility. Consider using
    the make_env function from the main envs module instead.
    
    Args:
        **kwargs: Arguments passed to SpaceMining constructor.
    
    Returns:
        Wrapped SpaceMining environment.
    """
    warnings.warn(
        "wrappers.make_env is deprecated. Use space_mining.envs.make_env instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    env = SpaceMining(**kwargs)
    env = apply_default_wrappers(env)
    return env


__all__ = [
    "FlattenActionSpaceWrapper",
    "ObservationNormalizationWrapper", 
    "RewardScalingWrapper",
    "FrameStackWrapper",
    "RecordingWrapper",
    "apply_default_wrappers",
    "make_env",
]
