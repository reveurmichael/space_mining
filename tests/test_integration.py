"""Integration tests for complete workflows involving renderer and GIF generation.

These tests simulate real usage patterns where users would:
1. Load an existing trained model
2. Generate visualizations/GIFs from the model
3. Test the complete pipeline from model to final output
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from space_mining.envs import make_env
from space_mining.agents.ppo_agent import PPOAgent
from space_mining.scripts.make_gif import generate_trajectory, save_gif
from space_mining.scripts.render_episode import render_episode


def test_complete_model_to_gif_workflow(shared_test_model, tmp_path):
    """Test the complete workflow from existing model to GIF generation.
    
    This simulates the real-world use case where a user has a trained model
    and wants to create a GIF visualization of the agent's behavior.
    """
    # Step 1: Generate trajectory from existing model
    frames = generate_trajectory(
        checkpoint_path=shared_test_model,
        num_steps=10,  # Short but meaningful trajectory
        render_mode="rgb_array",
        deterministic=True,
        device="cpu"
    )
    
    # Verify trajectory was generated
    assert len(frames) > 0
    # generate_trajectory produces num_steps + 1 frames (initial frame + step frames)
    assert len(frames) <= 11
    
    # Step 2: Create GIF from trajectory
    gif_path = tmp_path / "agent_behavior.gif"
    save_gif(frames, str(gif_path), fps=30)
    
    # Step 3: Verify GIF creation
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    
    # Step 4: Verify GIF properties
    with Image.open(gif_path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(frames)
        
    print(f"✅ Successfully created GIF with {len(frames)} frames at {gif_path}")


def test_model_loading_and_rendering_integration(shared_test_model):
    """Test loading a model and using it for rendering without saving GIF.
    
    This tests the integration between model loading and the renderer
    without the overhead of GIF creation.
    """
    # Load the model using PPOAgent
    env = make_env(render_mode="rgb_array", max_episode_steps=20)  # Short episode
    agent = PPOAgent.load(shared_test_model, env=env, device="cpu")
    
    # Reset environment and run a few steps
    obs, _ = env.reset()
    frames_collected = []
    
    for step in range(5):  # Just a few steps for speed
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and collect frame
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)
        frames_collected.append(frame)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Verify we collected frames and they're valid
    assert len(frames_collected) > 0
    assert all(isinstance(frame, np.ndarray) for frame in frames_collected)
    
    print(f"✅ Successfully rendered {len(frames_collected)} frames from model")

