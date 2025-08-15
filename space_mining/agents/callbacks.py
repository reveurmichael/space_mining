"""Robust callback wrappers for space_mining training.

This module provides WandbCallbackEveryN, a reliable wrapper around
wandb.integration.sb3.WandbCallback that logs at controlled intervals.
This keeps callback logic modular and separated from the main training script.
"""
from __future__ import annotations

from typing import Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallbackEveryN(BaseCallback):
    """Wrap wandb.integration.sb3.WandbCallback to log every N environment timesteps.

    This wrapper provides precise control over logging frequency while maintaining
    compatibility with Stable-Baselines3's callback system. It ensures that W&B
    logs are aligned with environment timesteps for accurate visualization.

    Parameters
    ----------
    log_every : int, default=1000
        Log to W&B every `log_every` environment timesteps. Must be positive.
    **wandb_kwargs : Any
        Passed directly to wandb.integration.sb3.WandbCallback constructor.
        Common parameters include:
        - model_save_path: Directory to save model checkpoints
        - model_save_freq: Frequency to save models (in timesteps)
        - gradient_save_freq: Frequency to log gradients (0 to disable)
        
    Attributes
    ----------
    log_every : int
        Logging frequency in environment timesteps.
    _wandb_callback : WandbCallback
        Internal WandbCallback instance.
    _last_logged_step : int
        Last environment timestep when logging occurred.

    Examples
    --------
    >>> # Basic usage
    >>> callback = WandbCallbackEveryN(log_every=1000)
    >>> 
    >>> # With model saving
    >>> callback = WandbCallbackEveryN(
    ...     log_every=1000,
    ...     model_save_path="./models",
    ...     model_save_freq=50000
    ... )
    
    Notes
    -----
    This callback automatically adds explicit step information to W&B logs
    to ensure proper x-axis alignment in the dashboard. This is crucial for
    having the correct number of data points (e.g., 10,000 points for 
    10M timesteps with log_every=1000).
    
    Raises
    ------
    ImportError
        If wandb or wandb.integration.sb3 is not available.
    ValueError
        If log_every is not positive.
    """

    def __init__(self, log_every: int = 1000, **wandb_kwargs: Any) -> None:
        super().__init__(verbose=0)
        
        if log_every <= 0:
            raise ValueError(f"log_every must be positive, got {log_every}")
            
        try:
            # Import here so module can be imported even if wandb is not installed
            from wandb.integration.sb3 import WandbCallback as _WandbCallback
        except ImportError as e:
            raise ImportError(
                "wandb and wandb.sb3 integration are required for WandbCallbackEveryN. "
                "Install with: pip install wandb"
            ) from e

        # Create the internal WandbCallback with passed kwargs
        # We don't set its model here; SB3 will call set_model on this wrapper
        self._wandb_callback = _WandbCallback(**wandb_kwargs)
        self.log_every = int(log_every)
        self._last_logged_step = 0

    def _on_training_start(self) -> None:
        """Initialize internal callback when training starts.
        
        This method ensures the internal WandbCallback is properly configured
        with the same model and training setup as this wrapper.
        """
        # Ensure internal callback sees the same model object
        try:
            # Preferred: call set_model if available
            if hasattr(self._wandb_callback, "set_model"):
                self._wandb_callback.set_model(self.model)
            else:
                # Fallback: set attribute directly
                self._wandb_callback.model = self.model
        except Exception as e:
            # Log warning but don't fail training
            if self.verbose >= 1:
                print(f"Warning: Failed to set model on internal callback: {e}")

        # Forward _on_training_start if present on internal callback
        if hasattr(self._wandb_callback, "_on_training_start"):
            try:
                self._wandb_callback._on_training_start()
            except Exception as e:
                # Be tolerant to internal callback errors
                if self.verbose >= 1:
                    print(f"Warning: Internal callback training start failed: {e}")

    def _on_step(self) -> bool:
        """Called by SB3 at each training step.
        
        This method implements the controlled logging logic, triggering the
        internal WandbCallback only when the specified number of environment
        timesteps have passed.
        
        Returns
        -------
        bool
            True to continue training, False to stop.
        """
        # Use model.num_timesteps (SB3 environment step counter)
        step = int(getattr(self.model, "num_timesteps", 0))

        if step - self._last_logged_step >= self.log_every:
            self._last_logged_step = step
            
            # Keep internal callback's model reference up-to-date
            try:
                self._wandb_callback.model = self.model
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Warning: Failed to update internal callback model: {e}")

            # Delegate to internal callback's _on_step if available
            if hasattr(self._wandb_callback, "_on_step"):
                try:
                    ok = bool(self._wandb_callback._on_step())
                except Exception as e:
                    # If internal callback fails, don't kill training
                    if self.verbose >= 1:
                        print(f"Warning: Internal callback step failed: {e}")
                    ok = True

                # Force W&B to log an entry with step parameter to ensure x-axis uses env steps
                try:
                    import wandb
                    # When there's an active run, write a lightweight metric (env_step) with explicit step
                    if wandb.run is not None:
                        # Use a small metric name, easy to hide or identify in dashboard
                        wandb.log({"env_step": int(step)}, step=int(step))
                except Exception as e:
                    # Never block training regardless of what happens
                    if self.verbose >= 1:
                        print(f"Warning: Failed to log env_step to W&B: {e}")

                return ok
        return True

    def _on_training_end(self) -> None:
        """Clean up when training ends.
        
        Forwards the training end event to the internal callback for proper cleanup.
        """
        if hasattr(self._wandb_callback, "_on_training_end"):
            try:
                self._wandb_callback._on_training_end()
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Warning: Internal callback training end failed: {e}")

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to internal WandbCallback.
        
        This allows the wrapper to act as a transparent proxy for any
        WandbCallback methods or attributes not explicitly implemented.
        
        Parameters
        ----------
        name : str
            Name of the attribute to access.
            
        Returns
        -------
        Any
            The requested attribute from the internal WandbCallback.
            
        Raises
        ------
        AttributeError
            If the attribute doesn't exist on the internal callback.
        """
        return getattr(self._wandb_callback, name)

    def __repr__(self) -> str:
        """String representation of the callback."""
        return f"WandbCallbackEveryN(log_every={self.log_every})"
