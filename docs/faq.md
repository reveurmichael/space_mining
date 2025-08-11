# FAQ

## What is SpaceMining?
A reinforcement-learning environment about asteroid mining with energy and obstacle management. See `environment_details.md`.

## Is it compatible with Gymnasium and SB3?
Yes. Import `space_mining` to auto-register, then `gym.make('SpaceMining')`. Training recipes are in `stable-baseline3.md` and `examples.md`.

## How do I install it?
Prefer `pip install space-mining`. More options in `installation.md`.

## How do I customize the environment?
Use `from space_mining import make_env` and pass kwargs (e.g., `max_episode_steps`, `observation_radius`). See `gymnasium.md` and `environment_details.md`.

## Which algorithm should I start with?
PPO. Starting hyperparameters and tuning notes are in `training_tips.md`.

## How long does training take?
Roughly 1–3M timesteps for strong performance, depending on settings and hardware.

## Rendering is slow or failing.
Use `render_mode='rgb_array'` during training. Ensure pygame/system SDL libraries are installed. See `installation.md` for platform tips.

## How do I record videos or GIFs?
See the GIF section and scripts in `examples.md`.

## "Environment 'SpaceMining' not found"
Import `space_mining` before `gym.make`, or call the factory `make_env()`. If issues persist, reinstall and check IDs.

## Does it support multi-agent?
Single-agent by default. Extend via subclassing if needed; contributions welcome.
