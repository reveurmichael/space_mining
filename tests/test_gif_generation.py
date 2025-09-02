"""Test GIF generation functionality and trajectory recording."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from PIL import Image

from space_mining.envs import make_env
from space_mining.agents.ppo_agent import PPOAgent
from space_mining.scripts.make_gif import save_gif, generate_trajectory
from space_mining import save_gif as save_gif_api


def create_test_model(tmp_path):
    """Create a minimal trained model for testing - reusable across tests."""
    env = make_env(render_mode="rgb_array")
    model = PPO("MlpPolicy", env, n_steps=2, verbose=0)
    model.learn(total_timesteps=10, progress_bar=False)
    
    model_path = tmp_path / "test_model.zip"
    model.save(model_path)
    env.close()
    
    return str(model_path)



def test_generate_trajectory_from_model(shared_test_model):
    """Test generating trajectory frames from a trained model."""
    # Use shared test model for speed
    
    # Generate short trajectory (fast)
    frames = generate_trajectory(
        checkpoint_path=shared_test_model,
        num_steps=5,  # Very short for speed
        render_mode="rgb_array",
        deterministic=True,
        device="cpu"
    )
    
    # Verify trajectory generation
    assert len(frames) > 0
    
    # Verify frame properties
    first_frame = frames[0]
    assert isinstance(first_frame, np.ndarray)
    assert first_frame.shape == (1080, 1920, 3)  # Updated resolution
    assert first_frame.dtype == np.uint8


def test_generate_trajectory_and_save_gif(shared_test_model, tmp_path):
    """Test complete workflow: generate trajectory and save as GIF."""
    # Use shared test model for speed
    
    # Generate trajectory
    frames = generate_trajectory(
        checkpoint_path=shared_test_model,
        num_steps=3,  # Minimal for speed
        render_mode="rgb_array",
        deterministic=True,
        device="cpu"
    )
    
    # Save as GIF
    gif_path = tmp_path / "trajectory.gif"
    save_gif(frames, str(gif_path), fps=30)
    
    # Verify complete workflow
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    
    # Verify GIF properties
    with Image.open(gif_path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(frames)


def test_trajectory_with_different_parameters(shared_test_model):
    """Test trajectory generation with different parameters."""
    # Use shared test model for speed
    
    # Test deterministic vs non-deterministic
    frames_det = generate_trajectory(shared_test_model, num_steps=2, deterministic=True)
    frames_stoch = generate_trajectory(shared_test_model, num_steps=2, deterministic=False)
    
    assert len(frames_det) > 0
    assert len(frames_stoch) > 0
    assert len(frames_det) == len(frames_stoch)


def test_agent_trajectory_integration(tmp_path):
    """Test integration between PPOAgent and trajectory generation."""
    # Create model using PPOAgent wrapper
    env = make_env(render_mode="rgb_array")
    agent = PPOAgent(env=env)
    
    # Minimal training
    agent.learn(total_timesteps=10, progress_bar=False)
    
    # Save model
    model_path = tmp_path / "agent_model.zip"
    agent.save(model_path)
    env.close()
    
    # Generate trajectory from saved agent model
    frames = generate_trajectory(
        checkpoint_path=str(model_path),
        num_steps=3,
        render_mode="rgb_array",
        deterministic=True
    )
    
    assert len(frames) > 0
    
    # Create GIF from agent trajectory
    gif_path = tmp_path / "agent_trajectory.gif"
    save_gif(frames, str(gif_path), fps=15)
    
    assert gif_path.exists()
