"""Small module containing WandbCallbackEveryN, a robust wrapper around
wandb.integration.sb3.WandbCallback that logs every N environment timesteps.

This keeps callback logic out of train_ppo.py.
"""
from __future__ import annotations

from typing import Any
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallbackEveryN(BaseCallback):
    """Wrap wandb.integration.sb3.WandbCallback and call it every N env steps.

    Parameters
    ----------
    log_every: int
        Log to W&B every `log_every` environment timesteps.
    **wandb_kwargs:
        Passed directly to wandb.integration.sb3.WandbCallback constructor
        (e.g. model_save_path, model_save_freq, gradient_save_freq).
    """

    def __init__(self, log_every: int = 1000, **wandb_kwargs: Any):
        super().__init__(verbose=0)
        try:
            # Import here so module can be imported even if wandb is not installed.
            from wandb.integration.sb3 import WandbCallback as _WandbCallback
        except Exception as e:
            raise ImportError(
                "wandb and wandb.sb3 integration are required for WandbCallbackEveryN"
            ) from e

        # Create the internal WandbCallback with passed kwargs.
        # We *don't* set its model here; SB3 will call set_model on this wrapper.
        self._wandb_callback = _WandbCallback(**wandb_kwargs)
        self.log_every = int(log_every)
        self._last_logged_step = 0

    def _on_training_start(self) -> None:
        """Bind the internal callback to the same model and forward start hook."""
        # Ensure internal callback sees the same model object
        try:
            # preferred: call set_model if available
            self._wandb_callback.set_model(self.model)
        except Exception:
            # fallback: set attribute directly
            self._wandb_callback.model = self.model

        # Forward _on_training_start if present on internal callback
        if hasattr(self._wandb_callback, "_on_training_start"):
            try:
                self._wandb_callback._on_training_start()
            except Exception:
                # be tolerant to internal callback errors
                pass

    def _on_step(self) -> bool:
        """Called by SB3 frequently (every step or at certain intervals)."""
        # Use model.num_timesteps (SB3 environment step counter)
        step = int(getattr(self.model, "num_timesteps", 0))

        if step - self._last_logged_step >= self.log_every:
            self._last_logged_step = step
            # Keep internal callback's model reference up-to-date
            try:
                self._wandb_callback.model = self.model
            except Exception:
                pass

            # Delegate to internal callback's _on_step if available.
            if hasattr(self._wandb_callback, "_on_step"):
                try:
                    ok = bool(self._wandb_callback._on_step())
                except Exception:
                    # if internal callback fails, don't kill training
                    ok = True

                # --- Force W&B to log an entry with step parameter to ensure x-axis uses env steps ---
                try:
                    import wandb
                    # When there's an active run, write a lightweight metric (env_step) with explicit step
                    if wandb.run is not None:
                        # Use a small metric name, easy to hide or identify in dashboard
                        wandb.log({"env_step": int(step)}, step=int(step))
                except Exception:
                    # Never block training regardless of what happens
                    pass

                return ok
        return True

    def _on_training_end(self) -> None:
        """Forward training end hook."""
        if hasattr(self._wandb_callback, "_on_training_end"):
            try:
                self._wandb_callback._on_training_end()
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to internal WandbCallback for anything we don't implement."""
        return getattr(self._wandb_callback, name)
