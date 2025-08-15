"""Test renderer functionality including visual effects and methods."""

import numpy as np
import pytest
from stable_baselines3 import PPO

from space_mining.envs import make_env
from space_mining.agents.ppo_agent import PPOAgent


def test_renderer_initialization():
    """Test that renderer initializes correctly with environment."""
    env = make_env(render_mode="rgb_array")
    assert env.unwrapped.renderer is not None
    assert hasattr(env.unwrapped.renderer, 'env')
    assert env.unwrapped.renderer.env is env.unwrapped
    env.close()


def test_renderer_cosmic_background():
    """Test that cosmic background initializes correctly."""
    env = make_env(render_mode="rgb_array")
    renderer = env.unwrapped.renderer
    
    # Check cosmic background elements are initialized
    assert hasattr(renderer, 'starfield_layers')
    assert hasattr(renderer, 'nebula_clouds')
    assert hasattr(renderer, 'distant_galaxies')
    assert hasattr(renderer, 'space_dust')
    assert hasattr(renderer, 'cosmic_auroras')
    
    # Check they contain elements
    assert len(renderer.starfield_layers) > 0
    assert len(renderer.nebula_clouds) > 0
    assert len(renderer.distant_galaxies) > 0
    assert len(renderer.space_dust) > 0
    assert len(renderer.cosmic_auroras) > 0
    
    env.close()


def test_renderer_visual_effects_methods():
    """Test that all visual effects methods exist and are callable."""
    env = make_env(render_mode="rgb_array")
    renderer = env.unwrapped.renderer
    
    # Test that all moved visualization methods exist
    methods_to_test = [
        'update_zoom',
        'update_event_timeline', 
        'update_combo_system',
        'spawn_delivery_particles',
        'add_score_popup',
        'add_timeline_event',
        'process_mining_combo',
        'update_animations',
        'trigger_game_over'
    ]
    
    for method_name in methods_to_test:
        assert hasattr(renderer, method_name), f"Method {method_name} not found in renderer"
        assert callable(getattr(renderer, method_name)), f"Method {method_name} is not callable"
    
    env.close()


def test_renderer_zoom_functionality():
    """Test zoom system functionality."""
    env = make_env(render_mode="rgb_array")
    renderer = env.unwrapped.renderer
    env.reset()
    
    # Test initial zoom values
    assert hasattr(env.unwrapped, 'zoom_level')
    assert hasattr(env.unwrapped, 'target_zoom')
    
    initial_zoom = env.unwrapped.zoom_level
    
    # Test zoom update
    renderer.update_zoom()
    
    # Zoom should stay within bounds
    assert 0.8 <= env.unwrapped.zoom_level <= 1.3
    
    env.close()


def test_renderer_animation_systems():
    """Test animation system components."""
    env = make_env(render_mode="rgb_array")
    renderer = env.unwrapped.renderer
    env.reset()
    
    # Test event timeline
    renderer.add_timeline_event("test", "Test Event", (255, 255, 255))
    assert len(env.unwrapped.event_timeline) == 1
    assert env.unwrapped.event_timeline[0]["text"] == "Test Event"
    
    # Test score popup
    test_pos = np.array([10.0, 10.0])
    renderer.add_score_popup("+10", test_pos, (255, 255, 0))
    assert len(env.unwrapped.score_popups) == 1
    assert env.unwrapped.score_popups[0]["text"] == "+10"
    
    # Test delivery particles
    start_pos = np.array([5.0, 5.0])
    target_pos = np.array([15.0, 15.0])
    renderer.spawn_delivery_particles(start_pos, target_pos)
    assert len(env.unwrapped.delivery_particles) == 10  # spawn_delivery_particles creates 10 particles
    
    env.close()


def test_renderer_combo_system():
    """Test combo system functionality."""
    env = make_env(render_mode="rgb_array")
    renderer = env.unwrapped.renderer
    env.reset()
    
    # Set up combo state
    env.unwrapped.combo_state["last_mining_step"] = env.unwrapped.steps_count
    
    # Test processing combo
    renderer.process_mining_combo()
    assert env.unwrapped.combo_state["chain_count"] >= 1
    
    # Test another combo in sequence
    env.unwrapped.steps_count += 10  # Small time gap
    renderer.process_mining_combo()
    assert env.unwrapped.combo_state["chain_count"] >= 2
    
    # Test combo display is triggered for 2+ combos
    if env.unwrapped.combo_state["chain_count"] >= 2:
        assert env.unwrapped.combo_state["display_timer"] > 0
    
    env.close()


def test_renderer_game_over_trigger():
    """Test game over screen functionality."""
    env = make_env(render_mode="rgb_array")
    renderer = env.unwrapped.renderer
    env.reset()
    
    # Set some basic stats for game over
    env.unwrapped.agent_inventory = 50.0
    env.unwrapped.collision_count = 5
    env.unwrapped.steps_count = 100
    
    # Trigger game over
    renderer.trigger_game_over(success=True)
    
    # Check game over state
    assert env.unwrapped.game_over_state["active"] is True
    assert env.unwrapped.game_over_state["success"] is True
    assert "final_stats" in env.unwrapped.game_over_state
    
    # Check final stats are populated
    stats = env.unwrapped.game_over_state["final_stats"]
    assert "current_inventory" in stats
    assert "collisions" in stats
    assert "steps_taken" in stats
    
    env.close()


def test_renderer_with_agent_episode():
    """Test renderer during a short episode with an agent."""
    env = make_env(render_mode="rgb_array")
    
    # Create a minimal trained model (very fast)
    model = PPO("MlpPolicy", env, n_steps=2, verbose=0)
    model.learn(total_timesteps=10, progress_bar=False)
    
    obs, _ = env.reset()
    
    # Run a few steps to test renderer in action
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Test that render returns valid frame
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)  # Updated resolution
        assert frame.dtype == np.uint8
        
        if terminated or truncated:
            break
    
    env.close()


def test_renderer_visual_elements():
    """Test that visual elements are properly rendered."""
    env = make_env(render_mode="rgb_array")
    env.reset()
    
    # Add some visual elements
    renderer = env.unwrapped.renderer
    renderer.add_timeline_event("mining", "+5.0", (255, 255, 0))
    renderer.add_score_popup("+10", np.array([20.0, 20.0]), (0, 255, 0))
    
    # Render frame with visual elements
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (1080, 1920, 3)
    
    # Test that the frame is not just black (has some visual content)
    assert np.sum(frame) > 0  # Frame should have some non-zero pixels
    
    env.close()


def test_renderer_performance():
    """Test that renderer performs reasonably well."""
    import time
    
    env = make_env(render_mode="rgb_array")
    env.reset()
    
    # Time multiple render calls
    start_time = time.time()
    num_renders = 10
    
    for _ in range(num_renders):
        frame = env.render()
        assert frame is not None
    
    end_time = time.time()
    avg_render_time = (end_time - start_time) / num_renders
    
    # Should render reasonably fast (less than 0.1s per frame in test environment)
    assert avg_render_time < 0.1, f"Rendering too slow: {avg_render_time:.3f}s per frame"
    
    env.close()