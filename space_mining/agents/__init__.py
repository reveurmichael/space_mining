"""Agents module for space_mining.

This module provides PPO agent implementations and training utilities for the
space_mining environment. It includes:

- PPOAgent: A wrapper class for PPO models with convenient loading/saving
- train_ppo: Main training function with configurable parameters
- WandbCallbackEveryN: Robust W&B logging callback with precise timing control

Examples:
    # Train a new agent
    >>> from space_mining.agents import train_ppo
    >>> model = train_ppo(total_timesteps=1000000, track_wandb=True)
    
    # Load and use a trained agent
    >>> from space_mining.agents import PPOAgent
    >>> agent = PPOAgent.load("./model.zip")
    >>> action, _ = agent.predict(observation)
"""

from .ppo_agent import PPOAgent
from .train_ppo import train_ppo, evaluate_trained_ppo
from .callbacks import WandbCallbackEveryN

__all__ = [
    "PPOAgent", 
    "train_ppo", 
    "evaluate_trained_ppo",
    "WandbCallbackEveryN"
]
