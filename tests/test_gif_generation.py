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


def test_save_gif_from_numpy_frames(tmp_path):
    """Test saving GIF from numpy array frames."""
    # Create test frames (small for speed)
    frames = []
    for i in range(3):  # Just 3 frames for speed
        # Create a simple colored frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 255  # Red, Green, Blue frames
        frames.append(frame)
    
    gif_path = tmp_path / "test_numpy.gif"
    
    # Test save_gif function
    save_gif(frames, str(gif_path), fps=10)
    
    # Verify GIF was created
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    
    # Verify GIF can be opened and has correct properties
    with Image.open(gif_path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 3


def test_save_gif_from_pil_frames(tmp_path):
    """Test saving GIF from PIL Image frames."""
    # Create test PIL frames
    frames = []
    for i in range(3):
        img = Image.new('RGB', (100, 100), color=(255 * (i == 0), 255 * (i == 1), 255 * (i == 2)))
        frames.append(img)
    
    gif_path = tmp_path / "test_pil.gif"
    
    # Test save_gif function
    save_gif(frames, str(gif_path), fps=15)
    
    # Verify GIF was created
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_save_gif_mixed_frames(tmp_path):
    """Test saving GIF from mixed numpy and PIL frames."""
    frames = []
    
    # Add numpy frame
    np_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    np_frame[:, :, 0] = 255  # Red
    frames.append(np_frame)
    
    # Add PIL frame
    pil_frame = Image.new('RGB', (50, 50), color=(0, 255, 0))  # Green
    frames.append(pil_frame)
    
    gif_path = tmp_path / "test_mixed.gif"
    save_gif(frames, str(gif_path), fps=5)
    
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_save_gif_api_wrapper(tmp_path, minimal_frames):
    """Test the top-level save_gif API wrapper."""
    gif_path = tmp_path / "test_api.gif"
    
    # Test API wrapper with minimal frames
    save_gif_api(minimal_frames, str(gif_path), fps=20)
    
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


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
    assert len(frames) <= 5  # Should not exceed requested steps
    
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


def test_gif_directory_creation(tmp_path):
    """Test that GIF creation automatically creates directories."""
    # Create nested directory path that doesn't exist
    nested_path = tmp_path / "subdir" / "nested" / "test.gif"
    
    # Create simple frame
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frames = [frame]
    
    # Should create directories automatically
    save_gif(frames, str(nested_path), fps=10)
    
    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_gif_fps_parameter():
    """Test that different FPS values work correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = [
            np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        # Test different FPS values
        for fps in [10, 30, 60]:
            gif_path = Path(tmp_dir) / f"test_{fps}fps.gif"
            save_gif(frames, str(gif_path), fps=fps)
            assert gif_path.exists()


def test_generate_trajectory_error_handling():
    """Test error handling in trajectory generation."""
    # Test with non-existent model path
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        generate_trajectory("non_existent_model.zip", num_steps=1)


def test_gif_with_single_frame(tmp_path):
    """Test GIF creation with a single frame."""
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :, 1] = 255  # Green frame
    
    gif_path = tmp_path / "single_frame.gif"
    save_gif([frame], str(gif_path), fps=1)
    
    assert gif_path.exists()
    
    with Image.open(gif_path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 1


def test_trajectory_episode_termination(shared_test_model):
    """Test that trajectory generation handles episode termination correctly."""
    # Use shared test model for speed
    
    # Request more steps than episode might last
    frames = generate_trajectory(
        checkpoint_path=shared_test_model,
        num_steps=100,  # More than likely episode length
        render_mode="rgb_array",
        deterministic=True,
        device="cpu"
    )
    
    # Should still generate some frames even if episode terminates early
    assert len(frames) > 0
    assert len(frames) <= 100  # Should not exceed requested


def test_gif_performance(tmp_path):
    """Test that GIF generation performs reasonably well."""
    import time
    
    # Create frames for performance test
    frames = [
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    gif_path = tmp_path / "performance_test.gif"
    
    # Time the GIF creation
    start_time = time.time()
    save_gif(frames, str(gif_path), fps=30)
    end_time = time.time()
    
    # Should complete quickly (less than 2 seconds for 5 small frames)
    creation_time = end_time - start_time
    assert creation_time < 2.0, f"GIF creation too slow: {creation_time:.3f}s"
    
    assert gif_path.exists()


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