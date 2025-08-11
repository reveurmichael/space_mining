"""Agents module for space_mining.
Exposes the PPOAgent class for training and inference.
"""

from .ppo_agent import PPOAgent
from .train_ppo import train_ppo

__all__ = ["PPOAgent", "train_ppo"]
