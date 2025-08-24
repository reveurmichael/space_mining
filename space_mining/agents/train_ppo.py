"""Training script for PPO agent in space_mining environment.
Can be run as a CLI script or imported as a function.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from space_mining.agents.callbacks import WandbCallbackEveryN

from space_mining import make_env


def _write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Write data to JSON file, creating directories as needed.
    
    Args:
        path: Path to write the JSON file.
        data: Data to serialize to JSON.
        
    Raises:
        OSError: If file creation fails.
        TypeError: If data is not JSON serializable.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except (OSError, TypeError) as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}") from e


def _resolve_value(val: Any) -> Union[float, str]:
    """Convert SB3 Schedule to float or handle other non-serializable values.
    
    Args:
        val: Value to resolve (could be a schedule, callable, or primitive).
        
    Returns:
        Resolved value as float or string representation.
    """
    if callable(val):
        try:
            return float(val(1.0))  # progress=1.0 (start of training)
        except Exception:
            return str(val)  # fallback: store repr
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def _gather_hyperparams_dict(model: PPO) -> Dict[str, Any]:
    """Extract key hyperparameters from the PPO instance.
    
    Args:
        model: Trained PPO model.
        
    Returns:
        Dictionary of hyperparameters with resolved values.
    """
    hp: Dict[str, Any] = {
        "policy": model.policy.__class__.__name__ if model.policy is not None else "MlpPolicy",
        "learning_rate": _resolve_value(getattr(model, "learning_rate", None)),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "gamma": getattr(model, "gamma", None),
        "gae_lambda": getattr(model, "gae_lambda", None),
        "clip_range": _resolve_value(getattr(model, "clip_range", None)),
        "ent_coef": getattr(model, "ent_coef", None),
        "vf_coef": getattr(model, "vf_coef", None),
        "max_grad_norm": getattr(model, "max_grad_norm", None),
        "device": str(getattr(model, "device", "cpu")),
    }
    return hp


def _gather_env_config_dict(env: Any) -> Dict[str, Any]:
    """Extract environment configuration parameters.
    
    Args:
        env: Environment instance (may be wrapped).
        
    Returns:
        Dictionary of environment configuration.
        
    Note:
        Uses env.unwrapped to access attributes directly from the base environment,
        avoiding Gymnasium v1.0+ deprecation warnings when accessing wrapped envs.
    """
    cfg: Dict[str, Any] = {}
    config_keys = (
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
    )
    
    for key in config_keys:
        # Use env.unwrapped to avoid deprecation warnings in Gymnasium v1.0+
        if hasattr(env.unwrapped, key):
            cfg[key] = getattr(env.unwrapped, key)
    return cfg


def _gather_training_args_dict(
    total_timesteps: int,
    device: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Gather training arguments and version information.
    
    Args:
        total_timesteps: Number of training timesteps.
        device: Training device.
        output_dir: Output directory path.
        
    Returns:
        Dictionary of training arguments and versions.
    """
    # Gather version information
    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split(" ")[0],
    }
    
    # Try to get package versions
    try:
        import gymnasium as gym  # type: ignore
        versions["gymnasium"] = getattr(gym, "__version__", None)
    except ImportError:
        versions["gymnasium"] = None
        
    try:
        import stable_baselines3 as sb3  # type: ignore
        versions["stable_baselines3"] = getattr(sb3, "__version__", None)
    except ImportError:
        versions["stable_baselines3"] = None
        
    try:
        import space_mining as sm  # type: ignore
        versions["space_mining"] = getattr(sm, "__version__", None)
    except ImportError:
        versions["space_mining"] = None

    return {
        "total_timesteps": int(total_timesteps),
        "device": device,
        "output_dir": output_dir,
        "versions": versions,
    }


def _write_training_artifacts(
    model: PPO,
    env: Any,
    total_timesteps: int,
    device: str,
    output_dir: str,
) -> None:
    """Write training artifacts (hyperparams, env config, training args) to JSON files.
    
    Args:
        model: Trained PPO model.
        env: Environment instance.
        total_timesteps: Number of training timesteps.
        device: Training device.
        output_dir: Output directory path.
    """
    # Gather all artifacts
    hyperparams = _gather_hyperparams_dict(model)
    env_config = _gather_env_config_dict(env)
    train_args = _gather_training_args_dict(
        total_timesteps=total_timesteps,
        device=device,
        output_dir=output_dir,
    )

    # Define artifact mapping
    artifacts = {
        "hyperparams.json": hyperparams,
        "env_config.json": env_config,
        "training_args.json": train_args,
    }

    # Write to both repo root and output_dir
    for filename, data in artifacts.items():
        # Write to repo root (for release workflow)
        _write_json(filename, data)
        # Write to output_dir (for convenience)
        _write_json(Path(output_dir) / filename, data)


def evaluate_trained_ppo(
    checkpoint_path: Union[str, Path],
    num_episodes: int = 10,
    device: str = "cpu",
    render_mode: Optional[str] = None,
    evaluation_json_path: Union[str, Path] = "evaluation.json",
) -> Dict[str, Any]:
    """Evaluate a trained PPO model and write metrics to a JSON file.

    Args:
        checkpoint_path: Path to the saved model file.
        num_episodes: Number of evaluation episodes.
        device: Device to use for evaluation.
        render_mode: Render mode for evaluation (None, 'human', 'rgb_array').
        evaluation_json_path: Path to write evaluation metrics.

    Returns:
        Dictionary with evaluation metrics: mean_reward, std_reward, episodes.
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If evaluation fails.
    """
    from space_mining.agents.ppo_agent import PPOAgent

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        env = make_env(render_mode=render_mode, max_episode_steps=1200)
        agent = PPOAgent.load(checkpoint_path, env=env, device=device)

        episode_rewards = []
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            
            while not (done or truncated):
                prediction = agent.predict(obs, deterministic=True)
                action = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
                obs, reward, done, truncated, _info = env.step(action)
                total_reward += float(reward)
                
            episode_rewards.append(total_reward)

        env.close()

        import numpy as np

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "episodes": int(len(episode_rewards)),
        }
        
        _write_json(evaluation_json_path, metrics)
        return metrics
        
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}") from e


def train_ppo(
    output_dir: str = "./runs/ppo",
    total_timesteps: int = 5_000_000,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    verbose: int = 1,
    render_mode: Optional[str] = None,
    device: str = "cpu",
    checkpoint_freq: int = 100000,
    eval_freq: int = 100000,
    track_wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    run_name: Optional[str] = None,
    log_every_n_env_steps: int = 1000,
) -> PPO:
    """Train a PPO model on the SpaceMining environment.

    Args:
        output_dir: Directory to save checkpoints and logs.
        total_timesteps: Total number of timesteps to train for.
        learning_rate: Learning rate for the PPO optimizer.
        n_steps: Number of steps to run per update.
        batch_size: Batch size for training.
        gamma: Discount factor.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        clip_range: Clipping parameter for PPO.
        verbose: Verbosity level.
        render_mode: Render mode for the environment (None, 'human', 'rgb_array').
        device: Device to use for training ('cpu' or 'cuda').
        checkpoint_freq: Save raw checkpoints every N steps (0 to disable).
        eval_freq: Evaluate and save best model every N steps (0 to disable).
        track_wandb: Enable Weights & Biases logging.
        wandb_project_name: W&B project name.
        wandb_entity: W&B entity (team or user).
        run_name: Run name for W&B.
        log_every_n_env_steps: Log every N environment timesteps.

    Returns:
        PPO: Trained PPO model.
        
    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If training fails.
    """
    # Validate parameters
    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
    if log_every_n_env_steps <= 0:
        raise ValueError(f"log_every_n_env_steps must be positive, got {log_every_n_env_steps}")
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"device must be 'cpu' or 'cuda', got '{device}'")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize Weights & Biases if enabled
    if track_wandb:
        try:
            import wandb  # type: ignore
            
            wandb.init(
                project=(wandb_project_name or "space-mining-ppo"),
                entity=(wandb_entity or "lundechen-shanghai-university"),
                name=run_name,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
        except ImportError as e:
            raise RuntimeError("wandb is required for W&B tracking. Install with: pip install wandb") from e

    try:
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
            tensorboard_log=str(output_path / "tensorboard_logs"),
            device=device,
        )

        # Build callbacks
        callbacks = []

        if track_wandb:
            # WandbCallback available only if wandb imports succeeded above
            callbacks.append(
                WandbCallbackEveryN(
                    log_every=log_every_n_env_steps,
                    model_save_path=str(output_path / "wandb_models"),
                    model_save_freq=checkpoint_freq if checkpoint_freq and checkpoint_freq > 0 else 0,
                    gradient_save_freq=0,
                )
            )

        if checkpoint_freq and checkpoint_freq > 0:
            checkpoint_dir = output_path / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix="ppo",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )

        if eval_freq and eval_freq > 0:
            eval_env = make_env(render_mode=None, max_episode_steps=1200)
            best_dir = output_path / "best_model"
            best_dir.mkdir(exist_ok=True)
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(output_path / "eval"),
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
        final_model_path = output_path / "final_model"
        model.save(str(final_model_path))
        print(f"Model saved to {final_model_path}")

        # Write training artifacts
        _write_training_artifacts(
            model=model,
            env=env,
            total_timesteps=total_timesteps,
            device=device,
            output_dir=output_dir,
        )

        # Finish the W&B run if enabled
        if track_wandb:
            wandb.finish()

        return model
        
    except Exception as e:
        # Clean up W&B if it was initialized
        if track_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
        raise RuntimeError(f"Training failed: {e}") from e


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on SpaceMining environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
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
        default=5_000_000,
        help="Total number of timesteps to train for",
    )
    
    # PPO hyperparameters
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
        "--batch-size", 
        dest="batch_size",
        type=int, 
        default=64, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.99, 
        help="Discount factor"
    )
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
    
    # Environment and system settings
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1, 
        help="Verbosity level"
    )
    parser.add_argument(
        "--render-mode",
        dest="render_mode",
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
        help="Device to use for training",
    )
    
    # Callback settings
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
    
    # W&B settings
    parser.add_argument(
        "--track-wandb",
        dest="track_wandb",
        action="store_true",
        help="Enable live logging to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        dest="wandb_project_name",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        dest="wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or user)",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        default=None,
        help="Run name for W&B",
    )
    parser.add_argument(
        "--log-every-n-env-steps",
        dest="log_every_n_env_steps",
        type=int,
        default=1000,
        help="Log every N environment timesteps",
    )

    args = parser.parse_args()

    try:
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
            track_wandb=args.track_wandb,
            wandb_project_name=args.wandb_project_name,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
            log_every_n_env_steps=args.log_every_n_env_steps,
        )
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
