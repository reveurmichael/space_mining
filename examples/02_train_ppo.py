"""Train a PPO agent on SpaceMining with sensible defaults.

This example calls the library-provided `train_ppo` utility for a 10M-step run
and writes artifacts to `runs/ppo/`.

Artifacts produced
- Periodic checkpoints (if configured in the utility)
- Tensorboard/W&B logs (depending on project settings)
- Final model file under `runs/ppo/`
"""

from space_mining.agents.train_ppo import train_ppo


def main() -> None:
    """Launch PPO training with default hyperparameters.

    Notes
    - Adjust `total_timesteps` and `output_dir` as needed for your experiments.
    - See the documentation for CLI equivalents and advanced settings.
    """
    train_ppo(total_timesteps=5_000_000, output_dir="runs/ppo")


if __name__ == "__main__":
    main()
