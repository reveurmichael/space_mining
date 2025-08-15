# SpaceMining PPO Agent

A PPO agent trained on the SpaceMining Gymnasium environment. This repository includes the final Stable-Baselines3 checkpoint, configuration, and evaluation metrics.

## Model Description

- Algorithm: PPO (Stable-Baselines3)
- Environment: SpaceMining (Gymnasium)
- Action Space: Box(3,) — thrust x, thrust y, mine toggle
- Observation Space: Box(53,) — agent state, nearby asteroids (up to 15), mothership relative position

## Quickstart

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO
from space_mining import make_env

ckpt_path = hf_hub_download(repo_id="LUNDECHEN/space-mining-ppo", filename="final_model.zip")
model = PPO.load(ckpt_path)

env = make_env(render_mode='rgb_array')
obs, _ = env.reset()
for _ in range(300):
    # SB3 `predict` may return `(action, state, *extras)` depending on version.
    prediction = model.predict(obs, deterministic=True)
    action = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## Training Configuration

- See `hyperparams.json` (algorithm hyperparameters)
- See `env_config.json` (environment parameters)
- See `training_args.json` (timesteps, device, versions)

## Evaluation

- See `evaluation.json`

| Metric        | Value |
|---------------|-------|
| mean_reward   | TBD   |
| std_reward    | TBD   |
| episodes      | TBD   |

## Agent Behavior

![Agent in action](agent_long.gif)

## License

- MIT 

## Authors

- Xinning Zhu (zhuxinning@shu.edu.cn)
- Lunde Chen (lundechen@shu.edu.cn)
