import numpy as np
from stable_baselines3 import PPO

from space_mining.agents.ppo_agent import PPOAgent
from space_mining.envs import make_env


def test_ppo_agent_predict_shape(tmp_path):
    env = make_env()
    model = PPO("MlpPolicy", env, n_steps=2)
    model_path = tmp_path / "test_model.zip"
    model.save(model_path)

    agent = PPOAgent.load(model_path)
    obs, _ = env.reset()
    action, _ = agent.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)

    env.close()