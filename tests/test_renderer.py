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
