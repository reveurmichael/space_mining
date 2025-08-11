"""Training script for PPO agent in space_mining environment.
Can be run as a CLI script or imported as a function.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from typing import Optional, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from space_mining import make_env


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _gather_hyperparams_dict(model: PPO) -> Dict[str, Any]:
    # Pull key hyperparameters from the PPO instance
    hp: Dict[str, Any] = {
        "policy": model.policy.__class__.__name__ if model.policy is not None else "MlpPolicy",
        "learning_rate": getattr(model, "learning_rate", None),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "gamma": getattr(model, "gamma", None),
        "gae_lambda": getattr(model, "gae_lambda", None),
        "clip_range": getattr(model, "clip_range", None),
        "ent_coef": getattr(model, "ent_coef", None),
        "vf_coef": getattr(model, "vf_coef", None),
        "max_grad_norm": getattr(model, "max_grad_norm", None),
        "device": str(getattr(model, "device", "cpu")),
    }
    return hp


def _gather_env_config_dict(env: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key in (
        "max_episode_steps",
        "grid_size",
        "max_asteroids",
        "max_resource_per_asteroid",
        "observation_radius",
        "max_obs_asteroids",
        "max_inventory",
        "mining_range",
        "dt",
        "mass",
        "max_force",
        "drag_coef",
        "gravity_strength",
        "obstacle_penalty",
        "energy_consumption_rate",
        "mining_energy_cost",
    ):
        if hasattr(env, key):
            cfg[key] = getattr(env, key)
    return cfg


def _gather_training_args_dict(
    total_timesteps: int,
    device: str,
    output_dir: str,
) -> Dict[str, Any]:
    try:
        import gymnasium as gym  # type: ignore
    except Exception:
        gym = None
    try:
        import stable_baselines3 as sb3  # type: ignore
    except Exception:
        sb3 = None
    try:
        import space_mining as sm  # type: ignore
    except Exception:
        sm = None

    versions: Dict[str, Any] = {
        "python": sys.version.split(" ")[0],
        "gymnasium": getattr(getattr(gym, "__version__", None), "__str__", lambda: None)() if gym else None,
        "stable_baselines3": getattr(sb3, "__version__", None) if sb3 else None,
        "space_mining": getattr(sm, "__version__", None) if sm else None,
    }

    return {
        "total_timesteps": int(total_timesteps),
        "device": device,
        "output_dir": output_dir,
        "versions": versions,
    }


def evaluate_trained_ppo(
    checkpoint_path: str,
    num_episodes: int = 10,
    device: str = "cpu",
    render_mode: Optional[str] = None,
    evaluation_json_path: str = "evaluation.json",
) -> Dict[str, Any]:
    """Evaluate a trained PPO model and write metrics to a JSON file.

    Returns the metrics dict with keys: mean_reward, std_reward, episodes.
    """
    from space_mining.agents.ppo_agent import PPOAgent

    env = make_env(render_mode=render_mode, max_episode_steps=1200)
    agent = PPOAgent.load(checkpoint_path, env=env, device=device)

    episode_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_r = 0.0
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = env.step(action)
            total_r += float(reward)
        episode_rewards.append(total_r)

    env.close()

    import numpy as np

    metrics = {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "episodes": int(len(episode_rewards)),
    }
    _write_json(evaluation_json_path, metrics)
    return metrics


def train_ppo(
    output_dir: str = "./runs/ppo",
    total_timesteps: int = 10_000_000,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    verbose: int = 1,
    render_mode: Optional[str] = None,
    device: str = "cpu",
    checkpoint_freq: int = 0,
    eval_freq: int = 0,
) -> PPO:
    """Train a PPO model on the SpaceMining environment.

    Args:
        output_dir (str): Directory to save checkpoints and logs.
        total_timesteps (int): Total number of timesteps to train for.
        learning_rate (float): Learning rate for the PPO optimizer.
        n_steps (int): Number of steps to run per update.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        gae_lambda (float): Lambda for Generalized Advantage Estimation.
        clip_range (float): Clipping parameter for PPO.
        verbose (int): Verbosity level.
        render_mode (str | None): Render mode for the environment (None, 'human', 'rgb_array').
        device (str): Device to use for training ('cpu' or 'cuda').
        checkpoint_freq (int): Save raw checkpoints every N steps (0 to disable).
        eval_freq (int): Evaluate and save best model every N steps (0 to disable).

    Returns:
        PPO: Trained PPO model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    env = make_env(render_mode=render_mode, max_episode_steps=1200)

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=verbose,
        tensorboard_log=os.path.join(output_dir, "tensorboard_logs"),
        device=device,
    )

    # Build callbacks
    callbacks = []

    if checkpoint_freq and checkpoint_freq > 0:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix="ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    if eval_freq and eval_freq > 0:
        eval_env = make_env(render_mode=None, max_episode_steps=1200)
        best_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=best_dir,
                log_path=os.path.join(output_dir, "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                render=False,
            )
        )

    callback_list = CallbackList(callbacks) if callbacks else None

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback_list)

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Dump JSON artifacts at repo root (as expected by release workflow) and in output_dir
    hyperparams = _gather_hyperparams_dict(model)
    env_config = _gather_env_config_dict(env)
    train_args = _gather_training_args_dict(
        total_timesteps=total_timesteps,
        device=device,
        output_dir=output_dir,
    )

    for rel in ("hyperparams.json", "env_config.json", "training_args.json"):
        # Write to repo root
        _write_json(rel, {
            "hyperparams": hyperparams,
            "env_config": env_config,
            "training_args": train_args,
        }[rel.replace(".json", "")] if rel != "hyperparams.json" else hyperparams)
        # Write to output_dir for convenience
        _write_json(os.path.join(output_dir, rel), {
            "hyperparams": hyperparams,
            "env_config": env_config,
            "training_args": train_args,
        }[rel.replace(".json", "")] if rel != "hyperparams.json" else hyperparams)

    return model


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on SpaceMining environment."
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./runs/ppo",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--total-timesteps",
        dest="total_timesteps",
        type=int,
        default=10000000,
        help="Total number of timesteps to train for",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=0.0003,
        help="Learning rate for PPO optimizer",
    )
    parser.add_argument(
        "--n-steps",
        dest="n_steps",
        type=int,
        default=2048,
        help="Number of steps to run per update",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--gae-lambda",
        dest="gae_lambda",
        type=float,
        default=0.95,
        help="Lambda for Generalized Advantage Estimation",
    )
    parser.add_argument(
        "--clip-range",
        dest="clip_range",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=["human", "rgb_array"],
        help="Render mode for the environment",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training (default: cpu)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        dest="checkpoint_freq",
        type=int,
        default=0,
        help="Save raw checkpoints every N steps (0 to disable)",
    )
    parser.add_argument(
        "--eval-freq",
        dest="eval_freq",
        type=int,
        default=0,
        help="Evaluate every N steps and save best model (0 to disable)",
    )

    args = parser.parse_args()

    train_ppo(
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=args.verbose,
        render_mode=args.render_mode,
        device=args.device,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    main()
