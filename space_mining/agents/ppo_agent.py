"""PPO Agent wrapper for space_mining.
Provides a simple interface for loading, predicting, and saving PPO models.
"""

from typing import Optional, Union, Any

from stable_baselines3 import PPO
import numpy as np


class PPOAgent:
    """A wrapper class for the PPO model used in space_mining."""

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
            env: The environment to train on (required for training when providing a policy name).
            device (str): Device to use ('cpu' or 'cuda', default: 'cpu').
            **kwargs: Additional arguments to pass to the PPO constructor when building a model.
        """
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

    @classmethod
    def load(cls, path: str, env: Optional[Any] = None, device: str = "cpu") -> "PPOAgent":
        """Load a trained PPO model from a file.

        Args:
            path (str): Path to the saved model file.
            env: The environment to associate with the model (if predicting).
            device (str): Device to use ('cpu' or 'cuda', default: 'cpu').

        Returns:
            PPOAgent: An instance of PPOAgent with the loaded model.
        """
        model: PPO = PPO.load(path, env=env, device=device)
        agent = cls(model, env=env, device=device)
        return agent

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
            filename: Model filename in the repo (default: "final_model.zip" or "best_model.zip").
            env: Optional environment to bind for inference.
            device: Device to use ('cpu' or 'cuda').
            token: Optional HF token if the repo is private.
        """
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # pragma: no cover - soft dependency
            raise ImportError(
                "huggingface_hub is required for load_from_hf(). Install with `pip install huggingface_hub`."
            ) from exc

        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return cls.load(local_path, env=env, device=device)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        """Predict an action given an observation.

        Args:
            observation: The current observation from the environment.
            deterministic (bool): Whether to use deterministic actions (default: True).

        Returns:
            action: The predicted action.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load a model or provide an environment.")
        return self.model.predict(observation, deterministic=deterministic)[0]

    def learn(self, total_timesteps: int, **kwargs: Any) -> "PPOAgent":
        """Train the PPO model.

        Args:
            total_timesteps (int): Total number of timesteps to train for.
            **kwargs: Additional arguments to pass to the learn method.

        Returns:
            self: The trained agent.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Provide an environment during initialization.")
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def save(self, path: str) -> None:
        """Save the PPO model to a file.

        Args:
            path (str): Path to save the model to.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot save.")
        self.model.save(path)
