# SpaceMining Examples (Short & Accurate)

Copy-paste snippets matching the current environment API and defaults.

## 1) Basic usage

```python
from space_mining.envs import SpaceMining

env = SpaceMining(render_mode=None)  # defaults: max_episode_steps=2000, grid_size=80, etc.
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

Or via factory:
```python
from space_mining import make_env
env = make_env(render_mode=None)
```

Common overrides (reflect actual defaults/limits):
```python
env = SpaceMining(
    max_episode_steps=2000,
    grid_size=80,
    max_asteroids=12,          # internally min-enforced to 18
    observation_radius=15,
    render_mode='human',       # or 'rgb_array' for headless
)
```

## 2) Train with Stable-Baselines3 (PPO)

Minimal script:
```python
from stable_baselines3 import PPO
from space_mining import make_env

env = make_env()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200_000)
model.save('ppo_space_mining.zip')
env.close()
```

Helper (Python API):
```python
from space_mining.agents.train_ppo import train_ppo
model = train_ppo(total_timesteps=5_000_000, output_dir='runs/ppo')
```

Helper (CLI):
```bash
python -m space_mining.agents.train_ppo --total-timesteps 5000000 --output-dir runs/ppo
```

## 3) Evaluate

```python
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from space_mining import make_env

env = make_env()
model = PPO('MlpPolicy', env).learn(10_000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(mean_reward, std_reward)
env.close()
```

## 4) Render and GIF

Render a checkpoint:
```bash
python -m space_mining.scripts.render_episode --model_path runs/ppo/final_model.zip
```

Generate GIF:
```bash
python -m space_mining.scripts.make_gif --checkpoint runs/ppo/final_model.zip --output output_gif/agent.gif --steps 1200 --fps 30 --deterministic
```

Python API:
```python
from space_mining.scripts.make_gif import generate_trajectory, save_gif
frames = generate_trajectory('runs/ppo/final_model.zip', num_steps=1200, deterministic=True)
save_gif(frames, 'output_gif/agent.gif', fps=30)
```

## 5) Load from Hugging Face

```python
from space_mining import make_env
from space_mining.agents.ppo_agent import PPOAgent

env = make_env(render_mode='rgb_array', max_episode_steps=2000)
agent = PPOAgent.load_from_hf('LUNDECHEN/space-mining-ppo', filename='final_model.zip', env=env)
obs, _ = env.reset()
for _ in range(600):
    action = agent.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
```

That’s all you need for quick experimentation. For details, see `docs/environment_details.md`.
These examples aim to be copy-paste friendly for users familiar with Gymnasium and Stable-Baselines3. If anything is unclear or you need additional scenarios, open an issue or PR. 🚀