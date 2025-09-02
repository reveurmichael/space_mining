"""Script to generate GIFs from SpaceMining environment trajectories or checkpoints.
Can be run as a CLI script or imported as a function.
"""

import argparse
import os
from typing import List, Sequence, Union

import numpy as np
from PIL import Image

from space_mining import make_env
from space_mining.agents.ppo_agent import PPOAgent


def save_gif(frames: Sequence[Union[np.ndarray, Image.Image]], output_path: str, fps: int = 30) -> None:
    """Save a sequence of frames as a GIF.

    Args:
        frames: List of frames (numpy arrays or PIL Images).
        output_path: Path to save the GIF file.
        fps: Frames per second for the GIF.
    """
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    pil_frames: List[Image.Image] = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames
    ]

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF saved to {output_path}")


def generate_trajectory(
    checkpoint_path: str,
    num_steps: int = 1200,
    render_mode: str = "rgb_array",
    deterministic: bool = True,
    device: str = "cpu",
) -> List[Union[np.ndarray, Image.Image]]:
    """Generate a trajectory from a checkpoint.

    Args:
        checkpoint_path: Path to the PPO checkpoint file.
        num_steps: Number of steps to run the trajectory for.
        render_mode: Render mode for the environment.
        deterministic: Whether to use deterministic predictions.
        device: Device to use ('cpu' or 'cuda', default: 'cpu').

    Returns:
        list: List of frames from the trajectory.
    """
    env = make_env(render_mode=render_mode, max_episode_steps=num_steps)

    agent = PPOAgent.load(checkpoint_path, env=env, device=device)

    obs, _ = env.reset()
    frames: List[Union[np.ndarray, Image.Image]] = [env.render()]

    for _ in range(num_steps):
        prediction = agent.predict(obs, deterministic=deterministic)
        # The Stable-Baselines3 API may return either a tuple (action, state, ...)
        # or a single ndarray depending on version and kwargs.  Extract the first
        # element if a sequence is returned so we remain version-agnostic.
        action = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()
    return frames


def main() -> None:
    """Main function to parse arguments and generate GIF."""
    parser = argparse.ArgumentParser(description="Generate a GIF from a SpaceMining PPO checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PPO checkpoint file")
    parser.add_argument("--output", type=str, default="output.gif", help="Path to save the output GIF")
    parser.add_argument(
        "--steps",
        dest="steps",
        type=int,
        default=1200,
        help="Number of steps to run for the trajectory",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the GIF")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic predictions (default: True)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)",
    )

    args = parser.parse_args()

    frames = generate_trajectory(
        checkpoint_path=args.checkpoint,
        num_steps=args.steps,
        deterministic=args.deterministic,
        device=args.device,
    )

    save_gif(frames, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
