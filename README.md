# SpaceMining — a novel RL environment beyond LLM priors

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/space-mining?color=6aa6ff)](https://pypi.org/project/space-mining/)
[![Tests](https://github.com/reveurmichael/space_mining/actions/workflows/run-pytest.yml/badge.svg)](https://github.com/reveurmichael/space_mining/actions/workflows/run-pytest.yml)
[![Docs](https://img.shields.io/badge/Docs-Site-2B3137)](https://reveurmichael.github.io/space_mining/docs/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-2B3137)](https://reveurmichael.github.io/space_mining/)
[![Release](https://img.shields.io/github/v/release/reveurmichael/space_mining?include_prereleases&color=6aa6ff)](https://github.com/reveurmichael/space_mining/releases)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reveurmichael/space_mining/blob/main/getting_started.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-space--mining--ppo-yellow?logo=huggingface)](https://huggingface.co/LUNDECHEN/space-mining-ppo)
[![W&B Project](https://img.shields.io/badge/W%26B-space--mining--ppo-fc4c02?logo=weightsandbiases)](https://wandb.ai/lundechen-shanghai-university/space-mining-ppo)

SpaceMining is a Gymnasium-compatible reinforcement learning (RL) environment designed to simulate asteroid mining in a 2D space. The agent, a mining robot, must collect resources from asteroids and deliver them to a central mothership while managing energy levels and avoiding moving obstacles. Featuring realistic physics, partial observability, and a comprehensive reward system, SpaceMining offers a challenging testbed for RL algorithms.

## What is SpaceMining?

SpaceMining is a single-agent reinforcement learning environment simulating asteroid mining in a 2D space environment. The agent (mining robot) must collect resources from asteroids, deliver them to the mothership, manage energy consumption, and avoid moving obstacles while maximizing efficiency.

## Research Purpose

This environment was specifically designed to evaluate large language models' ability to design reward functions for unfamiliar environments without prior knowledge. Recent studies have raised concerns that large language models may carry prior knowledge from pretraining data about standard RL environments (like CartPole, BipedalWalker, Ant), leading to potential prompt leakage and evaluation biases.

To address this issue, SpaceMining serves as a custom environment to assess true generalization capabilities on tasks free from such pretrained knowledge. This allows researchers to evaluate whether LLMs can effectively design reward functions for completely novel environments.

## Features

- 2D physics-based space mining simulation with realistic movement
- Dynamic resource management and energy consumption
- Partial observability with limited observation radius
- Moving obstacles and collision detection
- Customizable reward structure and environment parameters
- Real-time visualization with health bars and status indicators

## Task Description

The agent is deployed in a 2D space environment (80x80 grid) with randomly distributed asteroids and a central mothership. The agent must:

- Navigate efficiently to discover and mine resource-rich asteroids
- Manage energy consumption and return to the mothership for recharging
- Avoid collisions with moving obstacles
- Deliver resources to maximize collection efficiency
- Balance exploration and exploitation for optimal performance

The environment uses a comprehensive fitness scoring system with a target range of approximately 3000 points, evaluating resource collection, energy management, efficiency, and survival time.

## Environment Specifications

### State Space

The observation space includes:

- Agent State (6 dimensions): Position (x, y), velocity (vx, vy), energy level, inventory
- Asteroid Information (up to 45 dimensions): Relative positions and resource amounts for visible asteroids
- Mothership Position (2 dimensions): Relative position to mothership

### Action Space

The action space is continuous with 3 dimensions:

- Thrust Control (2 dimensions): Force applied in x and y directions [-1.0, 1.0]
- Mining Action (1 dimension): Binary mining activation [0.0, 1.0]

### Difficulty Level

Medium Difficulty: The environment presents a balanced challenge with:

- Limited observation radius (15 units) requiring strategic exploration
- Energy management constraints requiring periodic returns to mothership
- Moving obstacles requiring collision avoidance
- Resource depletion mechanics requiring efficient mining strategies

## Key Features

- **Gymnasium Compatibility**: Adheres to the latest Gymnasium API for seamless integration with standard RL workflows.
- **Stable-Baselines3 Support**: Optimized for training with PPO and other algorithms from Stable-Baselines3.
- **Complex Environment**: Includes energy management, partial observability (limited observation radius), continuous action spaces, and dynamic obstacles.
- **Visualization Tools**: Scripts to render episodes and generate GIFs for performance analysis.

## Installation

### From Source (Recommended for Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/reveurmichael/space_mining.git
   cd space_mining
   ```
2. Install in a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   pip install .
   ```
   For development mode:
   ```bash
   pip install -e '.[dev]'
   ```

### From PyPI (For Users)

Install from PyPI:
```bash
pip install space-mining
```

For detailed installation instructions and troubleshooting, see [Installation Guide](docs/installation.md).

## Getting Started

- Run the quickstart example:
  ```bash
  python examples/01_quickstart_random_agent.py
  ```
- Train a PPO agent:
  ```bash
  python -m space_mining.agents.train_ppo --total-timesteps 5000000 --output-dir runs/ppo
  ```
  Or, if you are in the root directory of the repository, you can use the following command:
  ```bash
  python space_mining/agents/train_ppo.py --total-timesteps 5000000 --output-dir runs/ppo
  ```
- Or load a pre-trained model from Hugging Face and render a GIF (no training):
  ```python
  from space_mining import make_env
  from space_mining.agents.ppo_agent import PPOAgent

  env = make_env(render_mode='rgb_array', max_episode_steps=1200)
  agent = PPOAgent.load_from_hf('LUNDECHEN/space-mining-ppo', filename='final_model.zip', env=env, device='cpu')
  obs, _ = env.reset()
  for _ in range(1200):
      action = agent.predict(obs, deterministic=True)
      obs, _, terminated, truncated, _ = env.step(action)
      if terminated or truncated:
          break
  env.close()
  ```
- Generate a GIF from a trained or HF checkpoint:
  ```bash
  python -m space_mining.scripts.make_gif --checkpoint runs/ppo/final_model.zip --output output_gif/agent.gif
  # or, if you are in the root directory of the repository, you can use the following command:
  python space_mining/scripts/make_gif.py --checkpoint runs/ppo/final_model.zip --output output_gif/agent.gif
  # or, after downloading from HF to a local file path
  python -m space_mining.scripts.make_gif --checkpoint path/to/final_model.zip --output output_gif/agent.gif
  ```

## Authors

- Xinning Zhu (zhuxinning@shu.edu.cn)
- Lunde Chen (lundechen@shu.edu.cn)
