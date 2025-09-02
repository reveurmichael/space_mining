"""PPO Agent wrapper for space_mining.
Provides a simple interface for loading, predicting, and saving PPO models.
"""

from typing import Optional, Union, Any, Tuple
from pathlib import Path

from stable_baselines3 import PPO
import numpy as np


class PPOAgent:
    """A wrapper class for the PPO model used in space_mining.
    
    This class provides a convenient interface for PPO model operations including
    loading from files or Hugging Face Hub, making predictions, training, and saving.
    
    Examples:
        # Create and train a new agent
        >>> from space_mining import make_env
        >>> env = make_env()
        >>> agent = PPOAgent("MlpPolicy", env=env)
        >>> agent.learn(total_timesteps=1000)
        
        # Load a pre-trained agent
        >>> agent = PPOAgent.load("./model.zip")
        >>> action, _ = agent.predict(observation)
        
        # Load from Hugging Face Hub
        >>> agent = PPOAgent.load_from_hf("LUNDECHEN/space-mining-ppo")
    """

    def __init__(
        self,
        policy_or_model: Union[str, PPO, None] = "MlpPolicy",
        env: Optional[Any] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the PPO agent.

        Args:
            policy_or_model: Either a policy name (e.g., 'MlpPolicy') or a prebuilt PPO model.
                If None, defaults to 'MlpPolicy'.
            env: The environment to train on (required for training when providing a policy name).
            device: Device to use ('cpu' or 'cuda', default: 'cpu').
            **kwargs: Additional arguments to pass to the PPO constructor when building a model.
            
        Raises:
            ValueError: If device is not 'cpu' or 'cuda'.
            ValueError: If policy_or_model is a string but env is None.
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got '{device}'")
            
        self.device: str = device
        self.model: Optional[PPO] = None
        self.policy: Optional[str] = None
        self.kwargs = dict(kwargs)

        if isinstance(policy_or_model, PPO):
            self.model = policy_or_model
            self.policy = None
        else:
            self.policy = policy_or_model if policy_or_model is not None else "MlpPolicy"
            if env is not None:
                self.kwargs["device"] = device
                self.model = PPO(self.policy, env, **self.kwargs)
            elif self.policy is not None:
                # Store policy for later initialization when env is provided
                pass

    @classmethod
    def load(cls, path: Union[str, Path], env: Optional[Any] = None, device: str = "cpu") -> "PPOAgent":
        """Load a trained PPO model from a file.

        Args:
            path: Path to the saved model file.
            env: The environment to associate with the model (if predicting).
            device: Device to use ('cpu' or 'cuda', default: 'cpu').

        Returns:
            PPOAgent: An instance of PPOAgent with the loaded model.
            
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If device is not 'cpu' or 'cuda'.
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got '{device}'")
            
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            model: PPO = PPO.load(str(path), env=env, device=device)
            agent = cls(model, env=env, device=device)
            return agent
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}") from e

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str,
        filename: str = "final_model.zip",
        env: Optional[Any] = None,
        device: str = "cpu",
        token: Optional[str] = None,
    ) -> "PPOAgent":
        """Load a trained PPO model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repo id, e.g., "LUNDECHEN/space-mining-ppo".
            filename: Model filename in the repo (default: "final_model.zip").
                Common alternatives include "best_model.zip".
            env: Optional environment to bind for inference.
            device: Device to use ('cpu' or 'cuda').
            token: Optional HF token if the repo is private.
            
        Returns:
            PPOAgent: An instance of PPOAgent with the loaded model.
            
        Raises:
            ImportError: If huggingface_hub is not installed.
            ValueError: If device is not 'cpu' or 'cuda'.
            RuntimeError: If loading from HF Hub fails.
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got '{device}'")
            
        if not repo_id.strip():
            raise ValueError("repo_id cannot be empty")
            
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise ImportError(
                "huggingface_hub is required for load_from_hf(). Install with `pip install huggingface_hub`."
            ) from exc

        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            return cls.load(local_path, env=env, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from HF Hub {repo_id}/{filename}: {e}") from e

    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict an action given an observation.

        Args:
            observation: The current observation from the environment.
            deterministic: Whether to use deterministic actions (default: True).
                If False, actions will be sampled stochastically.

        Returns:
            Tuple containing:
                - action: The predicted action.
                - state: The internal state (None for stateless policies).
                
        Raises:
            ValueError: If model is not initialized.
            RuntimeError: If prediction fails.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load a model or provide an environment during initialization.")
            
        try:
            # SB3's predict method can return different numbers of values depending on version
            # Handle both (action, state) and (action, state, log_prob) return formats
            result = self.model.predict(observation, deterministic=deterministic)
            if len(result) == 2:
                action, state = result
            elif len(result) == 3:
                action, state, _ = result  # Ignore log_prob
            else:
                # Fallback: take first two values
                action, state = result[0], result[1] if len(result) > 1 else None
            return action, state
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

    def learn(self, total_timesteps: int, **kwargs: Any) -> "PPOAgent":
        """Train the PPO model.

        Args:
            total_timesteps: Total number of timesteps to train for.
                Must be positive.
            **kwargs: Additional arguments to pass to the learn method.
                Common options include callback, log_interval, tb_log_name, etc.

        Returns:
            self: The trained agent for method chaining.
            
        Raises:
            ValueError: If model is not initialized or total_timesteps is invalid.
            RuntimeError: If training fails.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Provide an environment during initialization.")
            
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
            
        try:
            self.model.learn(total_timesteps=total_timesteps, **kwargs)
            return self
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}") from e

    def save(self, path: Union[str, Path]) -> None:
        """Save the PPO model to a file.

        Args:
            path: Path to save the model to. Directory will be created if it doesn't exist.
            
        Raises:
            ValueError: If model is not initialized.
            RuntimeError: If saving fails.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot save.")
            
        path_obj = Path(path)
        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {e}") from e

    def set_env(self, env: Any) -> None:
        """Set or update the environment for the agent.
        
        This is useful when you want to switch environments or when you 
        initialized the agent without an environment.
        
        Args:
            env: The new environment to associate with the model.
            
        Raises:
            ValueError: If model is not initialized.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot set environment.")
            
        self.model.set_env(env)

    @property
    def is_ready_for_prediction(self) -> bool:
        """Check if the agent is ready for making predictions.
        
        Returns:
            bool: True if the agent has a trained model ready for predictions.
        """
        return self.model is not None

    def __repr__(self) -> str:
        """String representation of the PPOAgent."""
        model_status = "initialized" if self.model is not None else "not initialized"
        return f"PPOAgent(device='{self.device}', model={model_status})"
