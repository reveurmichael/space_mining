# Space Mining: Extensive Examples for RL and GIF Generation

This document provides a wide range of examples for using the SpaceMining environment with Gymnasium and Stable-Baselines3 (SB3). It starts simple and gradually progresses to more advanced use cases, including training, evaluation, vectorized environments, custom policies, callbacks, hyperparameter tuning, rendering, and GIF generation.

- [0) Quick Setup](#0-quick-setup)
- [1) Basic Environment Usage (Gymnasium style)](#1-basic-environment-usage-gymnasium-style)
- [2) Basic Environment Usage (Factory function)](#2-basic-environment-usage-factory-function)
- [3) Training with Stable-Baselines3 (PPO)](#3-training-with-stable-baselines3-ppo)
- [4) PPOAgent convenience wrapper](#4-ppoagent-convenience-wrapper)
- [5) Vectorized Environments for Faster Training](#5-vectorized-environments-for-faster-training)
- [6) Evaluation](#6-evaluation)
- [7) Callbacks and Checkpointing](#7-callbacks-and-checkpointing)
- [8) Hyperparameter Tuning (Optuna sketch)](#8-hyperparameter-tuning-optuna-sketch)
- [9) Rendering Live Episodes](#9-rendering-live-episodes)
- [10) GIF Generation](#10-gif-generation)
- [11) Advanced: Recurrent Policies and Frame Stacking](#11-advanced-recurrent-policies-and-frame-stacking)
- [12) Environment Customization Ideas](#12-environment-customization-ideas)
- [13) Troubleshooting Quick Hits](#13-troubleshooting-quick-hits)
- [14) End-to-End Minimal Examples](#14-end-to-end-minimal-examples)


## 0) Quick Setup

- Install from source (recommended for development):

```bash
pip install -e '.[dev]'
```

- Or use `requirements.txt` directly:

```bash
pip install -r requirements.txt
```

- Import the package and make sure Gymnasium IDs are registered:

```python
from space_mining import register_envs
register_envs()
```


## 1) Basic Environment Usage (Gymnasium style)

```python
import gymnasium as gym
from space_mining import register_envs

# Register Gym ID
register_envs()

# Create by Gym ID
env = gym.make("SpaceMining")

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```


## 2) Basic Environment Usage (Factory function)

```python
from space_mining import make_env

env = make_env()  # same defaults as the Gym-registered env
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

- Common custom kwargs:
```python
env = make_env(
    render_mode=None,          # or 'human' or 'rgb_array'
    max_episode_steps=1200,
    grid_size=80,
    observation_radius=15,
)
```


## 3) Training with Stable-Baselines3 (PPO)

### 3.1 Minimal training script
```python
from stable_baselines3 import PPO
from space_mining import make_env

env = make_env()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200_000)
model.save('ppo_space_mining.zip')

env.close()
```

### 3.2 Use the built-in training helper
- CLI:
```bash
python -m space_mining.agents.train_ppo --total-timesteps 10000000 --output-dir runs/ppo
```
  - Use hyphenated flags only (e.g., `--total-timesteps`).

- Python API:
```python
from space_mining.agents.train_ppo import train_ppo

model = train_ppo(total_timesteps=10_000_000, output_dir='runs/ppo', device='cpu')
```




## 4) PPOAgent convenience wrapper

```python
from space_mining import PPOAgent, make_env

# Create and train
env = make_env()
agent = PPOAgent(env=env, device='cpu')
agent.learn(total_timesteps=10_000)
agent.save('agent_model.zip')

# Load and run
eval_env = make_env()
loaded = PPOAgent.load('agent_model.zip', env=eval_env, device='cpu')
obs, _ = eval_env.reset()
for _ in range(300):
    action = loaded.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
```

- You can also pass a prebuilt SB3 PPO model directly:
```python
from stable_baselines3 import PPO
from space_mining import PPOAgent, make_env

env = make_env()
model = PPO('MlpPolicy', env)
agent = PPOAgent(model, env=env)
```


## 5) Vectorized Environments for Faster Training

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from space_mining import make_env

n_envs = 8
env = SubprocVecEnv([lambda: make_env() for _ in range(n_envs)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10_000_000)
```


## 6) Evaluation

```python
from stable_baselines3.common.evaluation import evaluate_policy
from space_mining import make_env
from stable_baselines3 import PPO

env = make_env()
model = PPO('MlpPolicy', env).learn(10000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
env.close()
```


## 7) Callbacks and Checkpointing

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from space_mining import make_env
from stable_baselines3 import PPO

train_env = make_env()
eval_env = make_env()

model = PPO('MlpPolicy', train_env, verbose=1)
checkpoint = CheckpointCallback(save_freq=10_000, save_path='runs/ppo/', name_prefix='model')
eval_cb = EvalCallback(eval_env, best_model_save_path='runs/best/', log_path='runs/logs/', eval_freq=5_000,
                       deterministic=True, render=False)

model.learn(total_timesteps=10_000_000, callback=[checkpoint, eval_cb])
```


## 8) Hyperparameter Tuning (Optuna sketch)

```python
import optuna
from stable_baselines3 import PPO
from space_mining import make_env


def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 3e-3, log=True)
    n_steps = trial.suggest_int('n_steps', 512, 4096, step=512)

    env = make_env()
    model = PPO('MlpPolicy', env, learning_rate=lr, n_steps=n_steps, verbose=0)
    model.learn(total_timesteps=50_000)
    # Return a proxy metric; for real use, compute mean reward on eval episodes
    return float(model.ep_info_buffer[-1]['r']) if len(model.ep_info_buffer) else 0.0

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```


## 9) Rendering Live Episodes

- With a trained checkpoint:
```bash
python -m space_mining.scripts.render_episode --model_path runs/ppo/final_model.zip
```

- Programmatic:
```python
from space_mining.scripts.render_episode import render_episode

render_episode('runs/ppo/final_model.zip', max_steps=1200, device='cpu')
```


## 10) GIF Generation

### 10.0 Load a pre-trained model from Hugging Face (skip training)
```python
from space_mining import make_env
from space_mining.agents.ppo_agent import PPOAgent

env = make_env(render_mode='rgb_array', max_episode_steps=1200)
agent = PPOAgent.load_from_hf('LUNDECHEN/space-mining-ppo', filename='final_model.zip', env=env, device='cpu')
obs, _ = env.reset()
for _ in range(300):
    action = agent.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### 10.1 From the CLI
```bash
python -m space_mining.scripts.make_gif --checkpoint runs/ppo/final_model.zip --output output_gif/agent.gif --steps 1200 --fps 30 --deterministic
```
- `--steps` controls the number of frames rendered.

### 10.2 From Python API
```python
from space_mining.scripts.make_gif import generate_trajectory, save_gif

frames = generate_trajectory('runs/ppo/final_model.zip', num_steps=1200, deterministic=True)
save_gif(frames, 'output_gif/agent.gif', fps=30)
```

### 10.3 Custom overlays or sampling
```python
from space_mining.scripts.make_gif import generate_trajectory, save_gif

# Generate with stochastic actions for diversity
frames = generate_trajectory('runs/ppo/final_model.zip', num_steps=800, deterministic=False)
save_gif(frames, 'output_gif/agent_stochastic.gif', fps=24)
```


## 11) Advanced: Recurrent Policies and Frame Stacking

```python
from gymnasium.wrappers import FrameStack
from stable_baselines3 import PPO
from space_mining import make_env

stacked_env = FrameStack(make_env(), num_stack=4)
model = PPO('MlpPolicy', stacked_env, verbose=1)
model.learn(total_timesteps=500_000)
```


## 12) Environment Customization Ideas

- Make it easier:
```python
from space_mining import make_env

easy_env = make_env(observation_radius=20, grid_size=60, max_episode_steps=1500)
```

- Make it harder (smaller observation radius, more steps for planning):
```python
hard_env = make_env(observation_radius=10, grid_size=100, max_episode_steps=1800)
```


## 13) Troubleshooting Quick Hits

- Ensure Gymnasium/SB3 installed and compatible (see `requirements.txt`).
- On headless servers, use `render_mode='rgb_array'` and GIF generation instead of `'human'` rendering.
- If you hit ImportErrors for `pygame`, install it: `pip install pygame`.
- For CLI flags, both hyphen and underscore variants are supported where applicable.


## 14) End-to-End Minimal Examples

### 14.1 Train → Render → GIF
```bash
python -m space_mining.agents.train_ppo --total-timesteps 300000 --output-dir runs/quick
python -m space_mining.scripts.render_episode --model_path runs/quick/final_model.zip
python -m space_mining.scripts.make_gif --checkpoint runs/quick/final_model.zip --output output_gif/quick.gif --steps 600 --fps 30
```

### 14.2 Python-only
```python
from space_mining.agents.train_ppo import train_ppo
from space_mining.scripts.make_gif import generate_trajectory, save_gif

model = train_ppo(total_timesteps=300_000, output_dir='runs/quick')
frames = generate_trajectory('runs/quick/final_model.zip', num_steps=600, deterministic=True)
save_gif(frames, 'output_gif/quick.gif', fps=30)
```

---

These examples aim to be copy-paste friendly for users familiar with Gymnasium and Stable-Baselines3. If anything is unclear or you need additional scenarios, open an issue or PR. 🚀