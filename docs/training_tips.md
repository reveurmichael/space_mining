# Training Tips for SpaceMining

## What matters most

- **Partial observability**: limited sensing encourages memory or stacking
- **Energy management**: balance thrust/mining with timely recharges
- **Continuous control**: prefer algorithms suited for Box actions

## Recommended algorithms

- **PPO**: default, robust choice
- Also try **SAC** or **TD3** for continuous control

## Solid starting point (PPO)

- learning_rate: 3e-4
- n_steps: 2048, batch_size: 64, n_epochs: 10
- gamma: 0.99, gae_lambda: 0.95
- clip_range: 0.2, ent_coef: 0.01, vf_coef: 0.5
- max_grad_norm: 0.5

## Evaluate and monitor

- Periodically run `evaluate_policy` on a separate env
- Log reward components (from `info`) to see optimization focus
- Record occasional episodes in `'rgb_array'` and turn into GIFs

## Common issues

- **Not exploring**: increase `ent_coef` or exploration bonus; widen observation radius
- **Energy depletion**: lower consumption or increase recharge; adjust episode length
- **Unstable learning**: lower learning rate; increase batch size; tune clip range
- **Slow wall-clock**: add parallel envs; lower model size; fewer logging ops

For end-to-end scripts, callbacks, checkpointing, and GIF generation, see `examples.md`.
