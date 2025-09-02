"""Script to render an episode of the SpaceMining environment using a trained PPO model.
Can be run as a CLI script to visualize agent behavior.
"""
import argparse
from typing import Optional

from space_mining import PPOAgent, make_env


def render_episode(model_path: str, max_steps: int = 1200, device: str = "cpu") -> None:
    """Render an episode of the SpaceMining environment using a trained PPO model.

    Args:
        model_path (str): Path to the trained PPO model checkpoint.
        max_steps (int): Maximum number of steps to run the episode for.
        device (str): Device to use ('cpu' or 'cuda', default: 'cpu').
    """
    env = make_env(render_mode="human", max_episode_steps=max_steps)
    agent = PPOAgent.load(model_path, env=env, device=device)

    obs, _ = env.reset()

    for step_idx in range(max_steps):
        prediction = agent.predict(obs, deterministic=True)
        action = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break

    env.close()
    print(f"Episode rendering completed. Total steps: {min(max_steps, step_idx + 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render an episode of SpaceMining environment using a trained PPO model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained PPO model checkpoint",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1200,
        help="Maximum number of steps to run the episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)",
    )
    args = parser.parse_args()

    render_episode(args.model_path, args.max_steps, args.device)
