# Policy Gradient Homework — Developer Guide

## Overview

This repo implements **Parts 1–2** of the Policy Gradient homework:

| File | Purpose |
|------|---------|
| `1_simple_pg_gymnasium.py` | **Part 1a/1b** — Simple policy gradient on CartPole-v1, with optional visual rendering after each epoch |
| `2_reward_to_go_pg.py` | **Part 2** — Reward-to-go variant of the policy gradient |
| `run_experiments.py` | Runs both methods across multiple seeds and generates a comparison plot |
| `core.py` | Provided MLP Actor/Critic classes (used in Part 3+, not modified) |

---

## Environment Setup

```bash
# 1. Create the conda environment
conda create -n rl_env python=3.10 -y

# 2. Activate it
conda activate rl_env

# 3. Install dependencies
pip install torch gymnasium scipy matplotlib numpy
```

---

## File-by-File Explanation

### `1_simple_pg_gymnasium.py` (Part 1a + 1b)

**Original (Part 1a):** A basic REINFORCE-style policy gradient. Each step in an episode is weighted by the **total episode return** R(τ):

```
batch_weights += [ep_ret] * ep_len   # same weight for every step
```

**Added for Part 1b:** After each training epoch, a second environment with `render_mode="human"` runs 1 episode using the current policy (with `torch.no_grad()`) so you can visually watch the agent learn.

**Added for experiments:** `--seed`, `--epochs`, `--save_results` flags. When `--save_results` is set, per-epoch average returns are saved to `results/simple_pg_seed{N}.json`.

### `2_reward_to_go_pg.py` (Part 2)

A copy of the simple PG with one key change — the weighting. Instead of using the total episode return for every step, each step is weighted by its **reward-to-go** (sum of future rewards from that step onward):

```python
def reward_to_go(rews):
    # [r0, r1, ..., rT] -> [r0+r1+...+rT, r1+...+rT, ..., rT]
    rtgs = np.zeros_like(rews, dtype=np.float32)
    for i in reversed(range(len(rews))):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < len(rews) else 0)
    return rtgs
```

This reduces variance in the gradient estimate because early actions are no longer credited for rewards that happened before them.

### `run_experiments.py`

Orchestration script that:
1. Runs `1_simple_pg_gymnasium.py` N times with seeds 0..N-1
2. Runs `2_reward_to_go_pg.py` N times with seeds 0..N-1
3. Loads the saved JSON results
4. Plots **average return vs. epoch** (mean ± std shading) for both methods
5. Saves the plot as `comparison_plot.png`

---

## How to Reproduce

### Quick Test (verify everything works)

```bash
conda activate rl_env

# Test simple PG (5 epochs, ~10 seconds)
python 1_simple_pg_gymnasium.py --epochs 5

# Test reward-to-go PG (5 epochs, ~10 seconds)
python 2_reward_to_go_pg.py --epochs 5

# Test experiment pipeline (2 seeds × 10 epochs, ~2 minutes)
python run_experiments.py --epochs 10 --num_seeds 2
```

### Full Experiment (for the report)

```bash
conda activate rl_env

# Run 5 seeds × 50 epochs for both methods (~15-20 minutes)
python run_experiments.py --epochs 50 --num_seeds 5
```

This produces:
- `results/simple_pg_seed{0-4}.json` — per-epoch returns for simple PG
- `results/reward_to_go_seed{0-4}.json` — per-epoch returns for reward-to-go
- `comparison_plot.png` — the comparison figure for the report

### Visual Rendering (Part 1b)

```bash
# Watch the agent learn in real-time (opens a window each epoch)
python 1_simple_pg_gymnasium.py --epochs 50 --render
```

---

## CLI Reference

Both training scripts share the same interface:

| Flag | Default | Description |
|------|---------|-------------|
| `--env_name` | `CartPole-v1` | Gymnasium environment |
| `--lr` | `0.01` | Learning rate |
| `--epochs` | `50` | Number of training epochs |
| `--seed` | `0` | Random seed |
| `--render` | off | Enable visual rendering after each epoch |
| `--save_results` | off | Save per-epoch returns to `results/` |

`run_experiments.py` flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `50` | Epochs per run |
| `--num_seeds` | `5` | Number of random seeds |
| `--env_name` | `CartPole-v1` | Environment |
| `--skip_run` | off | Skip training, just plot from existing results |

---

## Output Structure

```
policy_gradient_homework/
├── 1_simple_pg_gymnasium.py    # Part 1a/1b
├── 2_reward_to_go_pg.py        # Part 2
├── run_experiments.py           # Experiment runner + plotting
├── core.py                      # MLP Actor/Critic (provided, for Part 3+)
├── comparison_plot.png          # Generated comparison figure
└── results/
    ├── simple_pg_seed0.json
    ├── simple_pg_seed1.json
    ├── ...
    ├── reward_to_go_seed0.json
    └── ...
```
