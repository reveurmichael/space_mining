import os

from stable_baselines3 import PPO

from space_mining.agents.ppo_agent import PPOAgent
from space_mining.envs import make_env


def test_ppo_agent_load_and_save(tmp_path):
    env = make_env()
    model = PPO("MlpPolicy", env, n_steps=2)
    model_path = tmp_path / "test_model.zip"
    model.save(model_path)

    agent = PPOAgent.load(model_path)
    assert agent.model is not None

    save_path = tmp_path / "saved_model.zip"
    agent.save(save_path)
    assert os.path.exists(save_path)

    env.close()