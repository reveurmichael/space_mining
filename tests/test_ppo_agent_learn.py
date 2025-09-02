import numpy as np

from space_mining.agents.ppo_agent import PPOAgent
from space_mining.envs import make_env


def test_ppo_agent_learn_and_reload(tmp_path):
    env = make_env()
    agent = PPOAgent(env=env)
    save_path = tmp_path / "trained_model.zip"

    agent.learn(total_timesteps=10, progress_bar=False)
    agent.save(save_path)

    loaded = PPOAgent.load(save_path, env=env)
    obs, _ = env.reset()
    action, _ = loaded.predict(obs)
    assert isinstance(action, np.ndarray)

    env.close()