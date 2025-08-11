# SpaceMining Environment Details

This document provides an in-depth look at the SpaceMining environment, including its observation space, action space, reward structure, and other key features.

## Overview

SpaceMining is a reinforcement learning environment designed to simulate asteroid mining in a 2D space. The agent (a mining robot) must collect resources from asteroids and deliver them to a central mothership while managing energy levels and avoiding moving obstacles. The environment features realistic physics simulation, partial observability, and a comprehensive reward system.

## Environment Specifications

### Observation Space

The observation space is a `Box` with shape `(53,)` and consists of the following components:

- **Agent State** (6 dimensions):
  - Position (x, y): 2 floats
  - Velocity (vx, vy): 2 floats
  - Energy level: 1 float (range: [0, 150])
  - Inventory level: 1 float (range: [0, 100])
- **Nearby Asteroids** (45 dimensions):
  - Up to 15 visible asteroids, each with:
    - Relative position (dx, dy): 2 floats
    - Resource amount: 1 float
  - If fewer than 15 asteroids are visible, the remaining values are zeros.
- **Mothership** (2 dimensions):
  - Relative position (dx, dy): 2 floats

Values are not normalized and reflect the actual positions and states within the environment grid (typically 80x80 units).

### Action Space

The action space is a `Box` with shape `(3,)` and range `[-1.0, 1.0]` for each dimension:

- `action[0]`: Thrust in the x-direction
- `action[1]`: Thrust in the y-direction
- `action[2]`: Mining activation (considered active if > 0.5)

These actions control the agent's movement and mining behavior.

### Reward Structure

The reward function in SpaceMining (following the GOODREWARD pattern) is designed to encourage efficient energy management, strategic exploration, optimal pathing, balanced speed control, and context-aware behavior. Key components include:

- **Speed Penalty**: Discourages excessive speed with a quadratic penalty if speed exceeds a limit.
- **Efficiency Reward**: Rewards energy conservation if energy ratio is above a threshold.
- **Exploration Reward**: Bonus for discovering new asteroids within the observation radius.
- **Path Efficiency Reward**: Encourages direct paths to objectives (mothership when carrying resources, asteroids when not).
- **Mining Guidance Reward**: Bonus for proximity to resource-rich asteroids when inventory is empty.
- **Delivery Guidance Reward**: Bonus for proximity to mothership when carrying resources or low on energy.

This reward structure complements immediate rewards from mining and delivery actions defined in the environment's step function.

### Termination and Truncation Conditions

- **Termination**: Occurs when the agent's energy is depleted, or if there are too many collisions with obstacles (threshold: 18 collisions).
- **Truncation**: Occurs when the maximum episode steps are reached (default: 1200 steps), or if all asteroids are depleted of resources.

### Environment Parameters

The environment can be customized with various parameters during initialization:

- `max_episode_steps`: Maximum steps per episode (default: 1200)
- `grid_size`: Size of the 2D space grid (default: 80)
- `max_asteroids`: Maximum number of asteroids (default: 12, minimum enforced: 18)
- `max_resource_per_asteroid`: Maximum resources per asteroid (default: 40)
- `observation_radius`: Radius within which the agent can observe asteroids (default: 15)
- `mining_range`: Range within which the agent can mine asteroids (default: 8.0)
- `max_inventory`: Maximum resources the agent can carry (default: 100)
- `render_mode`: Rendering mode ('human' for visual display, 'rgb_array' for image data, None for no rendering; default: None)

### Physics Simulation

The environment simulates 2D physics with the following elements:

- **Thrust and Acceleration**: Agent movement is controlled by thrust forces applied in x and y directions, scaled by a maximum force (default: 20.0).
- **Mass and Inertia**: Agent has a mass (default: 3.0) affecting acceleration.
- **Drag**: A drag coefficient (default: 0.02) slows down the agent over time.
- **Gravity**: A weak gravitational pull (default strength: 0.01) towards the mothership.
- **Boundary Conditions**: Agent is confined within grid boundaries with a margin, bouncing back with a penalty if it hits the edges.

### Energy Management

- **Consumption**: Energy is consumed at a base rate (default: 0.05 per step, reduced to half), with additional consumption for thrust and mining actions.
- **Recharge**: Energy is fully recharged to 150.0 when delivering resources to the mothership.
- **Depletion Penalty**: A penalty is applied if energy reaches zero, leading to episode termination.

### Resource Collection and Delivery

- **Mining**: Agent can mine resources from asteroids within the mining range (default: 8.0), with a high efficiency rate (60% of remaining resources per attempt).
- **Inventory**: Limited capacity (default: 100), with penalties for attempting to mine when full.
- **Delivery**: Resources are delivered to the mothership when within a certain range (default: 12.0), earning a significant reward.

## Rendering

The environment supports rendering in two modes:

- **Human Mode**: Displays a 2D top-down view using Pygame, showing the agent (green/yellow/orange circle), mothership (blue circle), asteroids (yellow circles with health bars), obstacles (red circles), and status information.
- **RGB Array Mode**: Returns frames as numpy arrays for processing or saving as videos/GIFs.

Visual elements include:
- **Agent**: Green (default), Yellow (carrying resources), Orange (mining)
- **Mothership**: Blue circle
- **Asteroids**: Yellow circles with health bars (gray X when depleted)
- **Obstacles**: Red circles
- **Ranges**: Blue ring (observation radius), Red ring (mining range)
- **Status Display**: Top-left text showing energy, inventory, total mined, collisions, steps, and asteroid count
- **Legend**: Bottom-right text explaining visual elements

## Fitness Score

A comprehensive fitness score is calculated to evaluate agent performance, targeting a range of approximately 3000 points for excellent performance. It considers:

- **Resources Collected**: Current inventory (weighted heavily)
- **Energy Remaining**: Normalized energy level
- **Resource Depletion Ratio**: Proportion of total resources collected
- **Proximity to Asteroids**: Bonus for being near resource-rich asteroids when not carrying resources
- **Proximity to Mothership**: Bonus for being near mothership when carrying resources
- **Completion Bonus**: For depleting all asteroids
- **Efficiency Bonus**: For resource collection efficiency per step
- **Survival Bonus**: For episode duration

## Customization

You can subclass `SpaceMining` to modify the reward function, physics parameters, or add new features like multi-agent support. Ensure to update the observation and action spaces accordingly if changes are made.

## Usage Example

```python
from space_mining import make_env

# Create environment with custom parameters
env = make_env(
    max_episode_steps=1500,
    grid_size=100,
    max_asteroids=15,
    observation_radius=20,
    render_mode='human'
)

# Reset and run a few steps
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action for demo
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

This document should provide a thorough understanding of the SpaceMining environment's mechanics and how to interact with it effectively in reinforcement learning tasks.
