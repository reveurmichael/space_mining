# Using SpaceMining with Stable-Baselines3

## Minimal PPO training

```python
from stable_baselines3 import PPO
from space_mining import make_env

env = make_env()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=2_000_000)
model.save('ppo_space_mining.zip')
env.close()
```

## Evaluate

```python
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
```

## Save / load

```python
from stable_baselines3 import PPO
model = PPO.load('ppo_space_mining.zip')
```

## Starting hyperparameters (PPO)

- learning_rate: 3e-4
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99, gae_lambda: 0.95
- clip_range: 0.2, ent_coef: 0.01, vf_coef: 0.5, max_grad_norm: 0.5

## Vectorized training (faster)

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from space_mining import make_env

n_envs = 8
env = SubprocVecEnv([lambda: make_env() for _ in range(n_envs)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=5_000_000)
```

- More recipes (callbacks, checkpoints, HF loading, GIFs): see `examples.md`.
- Tuning guidance and common pitfalls: see `training_tips.md`.
