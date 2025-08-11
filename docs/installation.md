# Installation

- Requirements: Python 3.10+, pip
- Recommended: use a virtual environment

## Option A: PyPI (fastest)

```bash
pip install space-mining
```

## Option B: From source (for development)

```bash
git clone https://github.com/reveurmichael/space_mining.git
cd space_mining
python -m venv venv && source venv/bin/activate
pip install -e '.[dev]'
```

## Verify

```python
import gymnasium as gym
import space_mining  # registers the env

env = gym.make('SpaceMining')
obs, info = env.reset()
print('OK')
env.close()
```
