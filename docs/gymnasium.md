# Using SpaceMining with Gymnasium

SpaceMining follows the Gymnasium API and auto-registers on import.

## Quickstart

```python
import gymnasium as gym
import space_mining  # auto-registers 'SpaceMining'

env = gym.make('SpaceMining')
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Customize creation

Prefer the convenience factory for kwargs:

```python
from space_mining import make_env

env = make_env(
    render_mode=None,      # or 'human' / 'rgb_array'
    max_episode_steps=1200,
    grid_size=80,
    observation_radius=15,
)
```

## Vectorized environments

```python
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym
import space_mining

def make_sm():
    return gym.make('SpaceMining')

vec_env = SyncVectorEnv([make_sm for _ in range(4)])
```

- For training recipes, see `stable-baseline3.md` and `training_tips.md`.
- For more examples (evaluation, callbacks, GIFs), see `examples.md`.
- For full environment specs, see `environment_details.md`.
