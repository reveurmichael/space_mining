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
    assert len(frames) <= 10
    
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


def test_renderer_visual_effects_with_model(shared_test_model):
    """Test that visual effects work correctly during model execution."""
    env = make_env(render_mode="rgb_array", max_episode_steps=30)
    agent = PPOAgent.load(shared_test_model, env=env, device="cpu")
    
    obs, _ = env.reset()
    renderer = env.unwrapped.renderer
    
    # Run a few steps to potentially trigger visual effects
    effects_triggered = {
        "mining": False,
        "delivery": False,
        "collision": False,
        "animations": False
    }
    
    for step in range(10):  # Short run
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if various visual effects are active
        if len(env.unwrapped.agent_trail) > 0:
            effects_triggered["animations"] = True
            
        if len(env.unwrapped.event_timeline) > 0:
            for event in env.unwrapped.event_timeline:
                if event["type"] == "mining":
                    effects_triggered["mining"] = True
                elif event["type"] == "delivery":
                    effects_triggered["delivery"] = True
                elif event["type"] == "collision":
                    effects_triggered["collision"] = True
        
        # Render to ensure no errors during visual effects
        frame = env.render()
        assert frame is not None
        
        if terminated or truncated:
            break
    
    env.close()
    
    # At minimum, animations should be active (agent trail)
    assert effects_triggered["animations"], "Agent trail animations should be active"
    
    print(f"✅ Visual effects tested: {effects_triggered}")


def test_gif_quality_and_size_optimization(shared_test_model, tmp_path):
    """Test different GIF parameters for quality and file size."""
    # Generate a short trajectory
    frames = generate_trajectory(
        checkpoint_path=shared_test_model,
        num_steps=5,
        render_mode="rgb_array",
        deterministic=True,
        device="cpu"
    )
    
    # Test different FPS settings and measure file sizes
    fps_tests = [10, 30, 60]
    file_sizes = {}
    
    for fps in fps_tests:
        gif_path = tmp_path / f"test_{fps}fps.gif"
        save_gif(frames, str(gif_path), fps=fps)
        
        assert gif_path.exists()
        file_sizes[fps] = gif_path.stat().st_size
        
        # Verify GIF properties
        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames == len(frames)
    
    # All files should be created and have reasonable sizes
    assert all(size > 1000 for size in file_sizes.values()), "GIF files should be reasonably sized"
    
    print(f"✅ GIF file sizes at different FPS: {file_sizes}")


def test_error_recovery_in_gif_generation(tmp_path):
    """Test that GIF generation handles errors gracefully."""
    # Test with empty frames list
    with pytest.raises((ValueError, IndexError)):
        save_gif([], str(tmp_path / "empty.gif"), fps=30)
    
    # Test with invalid frames
    invalid_frames = [np.array([1, 2, 3])]  # Wrong shape
    with pytest.raises((ValueError, TypeError)):
        save_gif(invalid_frames, str(tmp_path / "invalid.gif"), fps=30)
    
    # Test with non-existent model path
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        generate_trajectory("non_existent_model.zip", num_steps=1)
    
    print("✅ Error handling tests passed")


def test_memory_usage_optimization():
    """Test that rendering and GIF generation don't consume excessive memory."""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create environment and generate some frames
    env = make_env(render_mode="rgb_array", max_episode_steps=10)
    frames = []
    
    obs, _ = env.reset()
    for _ in range(5):  # Short sequence
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Create GIF
    with tempfile.TemporaryDirectory() as tmp_dir:
        gif_path = Path(tmp_dir) / "memory_test.gif"
        save_gif(frames, str(gif_path), fps=30)
    
    # Check final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 500MB for this test)
    assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
    
    print(f"✅ Memory usage test passed (increase: {memory_increase:.1f}MB)")


def test_concurrent_gif_generation(shared_test_model, tmp_path):
    """Test that multiple GIF generations can work without interference."""
    import threading
    import queue
    
    results = queue.Queue()
    
    def generate_gif(gif_name):
        try:
            # Generate trajectory
            frames = generate_trajectory(
                checkpoint_path=shared_test_model,
                num_steps=3,  # Very short for speed
                render_mode="rgb_array",
                deterministic=True,
                device="cpu"
            )
            
            # Save GIF
            gif_path = tmp_path / f"{gif_name}.gif"
            save_gif(frames, str(gif_path), fps=15)
            
            results.put(("success", gif_name, gif_path.exists()))
        except Exception as e:
            results.put(("error", gif_name, str(e)))
    
    # Start multiple threads
    threads = []
    for i in range(3):  # Test with 3 concurrent generations
        thread = threading.Thread(target=generate_gif, args=(f"concurrent_{i}",))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Check results
    successes = 0
    while not results.empty():
        status, name, result = results.get()
        if status == "success":
            assert result is True, f"GIF {name} was not created successfully"
            successes += 1
        else:
            pytest.fail(f"Thread {name} failed: {result}")
    
    assert successes == 3, "All concurrent GIF generations should succeed"
    print("✅ Concurrent GIF generation test passed")