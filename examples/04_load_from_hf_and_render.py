"""Load a pre-trained agent from Hugging Face and render an episode.

This example demonstrates how to use `PPOAgent.load_from_hf` to fetch a
public model and evaluate it in the environment, rendering frames live.
"""

from space_mining import make_env
from space_mining.agents.ppo_agent import PPOAgent


def main() -> None:
    """Load the HF agent and render a short evaluation episode.

    Steps
    1. Make the environment in human render mode for a fast visual demo.
    2. Load the `final_model.zip` from the specified HF repo using CPU.
    3. Step the environment using the agent's greedy action for ~600 steps.
    4. Stop early if the episode terminates or truncates; always close env.
    """
    env = make_env(render_mode="human", max_episode_steps=600)
    agent = PPOAgent.load_from_hf("LUNDECHEN/space-mining-ppo", filename="final_model.zip", env=env, device="cpu")

    obs, _ = env.reset()
    for _ in range(600):
        prediction = agent.predict(obs, deterministic=True)
        action = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()