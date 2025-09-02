# Reproducibility Guide

This project provides a single GitHub Actions workflow that trains agents and publishes results so others can reproduce your experiments and figures.

## Unified CI Run (W&B + HF)
Use the "Training to W&B + HF" workflow for both short smoke checks and long publication runs by adjusting inputs.
- Inputs: `total_timesteps`, `checkpoint_freq`, `eval_freq`, `run_name`
- Outputs:
  - W&B: Synced logs, model artifacts (best/final), and GIF
  - Hugging Face Hub: uploaded best/final model `.zip` and demo GIF
  - GitHub Artifacts: GIF

Secrets/Variables required:
- `WANDB_API_KEY` (secret)
- `HF_TOKEN` (secret) with write access

## Local Reproduction
- Install dependencies: `pip install -r requirements.txt`
- Train: `python -m space_mining.agents.train_ppo --total-timesteps 5000000 --output-dir runs/ppo`
- Generate GIF: `python space_mining/scripts/make_gif.py --checkpoint runs/ppo/final_model.zip --output output_gif/agent.gif --steps 800 --fps 20`
