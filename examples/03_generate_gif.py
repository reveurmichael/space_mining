"""Generate and save a demonstration GIF from a trained PPO checkpoint.

This script uses the high-level helpers `generate_trajectory` and `save_gif`
from `space_mining.scripts.make_gif` to render an episode and persist it.

Typical uses
- Produce publication-ready demos for reports or READMEs
- Visually inspect agent behaviors at different training stages
"""

from space_mining.scripts.make_gif import generate_trajectory, save_gif


def main() -> None:
    """Render a trajectory and save it as a GIF.

    Behavior
    - Loads a checkpoint from `runs/ppo/final_model.zip`
    - Runs for 1200 steps deterministically
    - Writes a 30 FPS GIF to `output_gif/agent.gif`
    """
    checkpoint = "runs/ppo/final_model.zip"
    frames = generate_trajectory(
        checkpoint_path=checkpoint, num_steps=1200, deterministic=True
    )
    save_gif(frames, "output_gif/agent.gif", fps=30)


if __name__ == "__main__":
    main()
