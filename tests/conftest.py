"""Shared pytest fixtures for space_mining tests."""

import pytest
import tempfile
from pathlib import Path

from stable_baselines3 import PPO
from space_mining.envs import make_env


@pytest.fixture(scope="session")
def shared_test_model():
    """Create a shared trained model for tests that need one.
    
    Using session scope to create it once per test session to save time.
    Returns a temporary file path that will be cleaned up automatically.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = make_env(render_mode="rgb_array")
        model = PPO("MlpPolicy", env, n_steps=2, verbose=0)
        model.learn(total_timesteps=20, progress_bar=False)  # Slightly more training for stability
        
        model_path = Path(tmp_dir) / "shared_test_model.zip"
        model.save(model_path)
        env.close()
        
        yield str(model_path)
        # Cleanup happens automatically when tmp_dir context exits


@pytest.fixture
def fast_env():
    """Create a fast environment for testing."""
    env = make_env(render_mode="rgb_array", max_episode_steps=50)  # Short episodes for speed
    yield env
    env.close()


@pytest.fixture
def minimal_frames():
    """Create minimal test frames for GIF testing."""
    import numpy as np
    
    frames = []
    for i in range(3):  # Just 3 frames
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 128  # Different colors but not too bright
        frames.append(frame)
    
    return frames