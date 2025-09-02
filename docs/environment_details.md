# SpaceMining: Key Environment Details

Concise reference aligned with `space_mining/envs/space_mining.py`.

## Spaces

- Observation: `Box(shape=(53,), dtype=float32)`
  - Agent (6): position (2), velocity (2), energy, inventory
  - Asteroids (45): up to 15 × [rel_x, rel_y, resources]
  - Mothership (2): relative position (dx, dy)
- Action: `Box(shape=(3,), low=[-1,-1,0], high=[1,1,1])`
  - `[thrust_x, thrust_y, mine_action]` (mine active if > 0.5)

## Defaults (core params)

- `max_episode_steps=2000`
- `grid_size=80`
- `max_asteroids=12` (internally enforced minimum: 18)
- `max_resource_per_asteroid=40`
- `observation_radius=15`
- `mining_range=8.0`
- `max_inventory=100`
- Physics: `dt=0.1`, `mass=3.0`, `max_force=20.0`, `drag_coef=0.02`, `gravity_strength=0.01`
- Energy: base `0.05`/step, extra for thrust (0.01 × |thrust| sum) and mining (0.5 when mining)
- Render: `render_modes=["human","rgb_array"]`, `render_fps=30`

## Episode flow

1. Reset spawns 8–12 asteroids (each 25–40 resources), 4–8 moving obstacles, full energy 150, inventory 0.
2. Step applies thrust, drag, weak gravity; handles mining within 8.0 units; delivery within 12.0 units fully recharges energy to 150.
3. Termination:
   - Energy <= 0
   - Too many collisions (>= 8)
   - All asteroids depleted (exploration complete)
   - Or time limit (`max_episode_steps`) → truncation

## Rewards (as implemented)

- Mining reward: +50 per mining_success
- Delivery reward: +100 per delivery_success
- Penalties: -20 per obstacle collision; -10 on boundary collision; -0.1 × (energy - 10.0)
- Time step reward: +0.01

## Fitness score (info)

Combines inventory, normalized energy, resource depletion ratio, proximity to asteroids/mothership, completion, efficiency, and survival bonuses to target high-performing runs.
