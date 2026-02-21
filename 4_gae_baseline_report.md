# Part 4: GAE + Value Function Baseline - Report

## Overview

This report documents the implementation of **Generalized Advantage Estimation (GAE)** with a **learned value function baseline**. The policy gradient update uses advantage estimates as weights instead of raw reward-to-go, and the value function is trained with MSE on the actual returns (reward-to-go from each state). This reduces variance and typically improves sample efficiency and stability compared to the reward-to-go formulation without a baseline (Part 3).

## Implementation Details

### Algorithm

1. **Data collection**: Same on-policy batch collection as before. For each step we store observation, action, reward, value estimate \(V(s)\), and log probability \(\log \pi(a|s)\) using `MLPActorCritic.step()`.

2. **GAE (Generalized Advantage Estimation)**  
   At the end of each episode we compute TD errors and then the GAE(\(\gamma\), \(\lambda\)) advantage:
   - \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\) (with \(V(s_T)=0\) when the episode ends)
   - \(\hat{A}_t^{\text{GAE}} = \sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l}\), implemented via `discount_cumsum(deltas, gamma * lam)` from `core.py`.

3. **Value targets**: The target for the value function at each state is the **reward-to-go** from that state (discounted sum of future rewards), i.e. the same quantity we used as weights in Part 2/3, now used as regression targets for \(V(s)\).

4. **Normalization**: Advantage estimates are normalized to mean 0 and standard deviation 1 (as in the Spinning Up VPG implementation) before the policy update to improve stability.

5. **Updates**:
   - **Policy**: One gradient step minimizing \(-\mathbb{E}[\log \pi(a|s) \cdot \hat{A}]\) (policy gradient weighted by advantage).
   - **Value function**: Multiple gradient steps (default `train_v_iters=80`) minimizing \(\mathbb{E}[(V(s) - \text{ret})^2]\) (MSE between predicted value and actual return). Using several value steps per policy step improves stability.

### Code Structure

- **`4_gae_baseline_pg.py`**:
  - `gae_and_returns(rewards, values, last_val, gamma, lam)`: Computes GAE advantages and return targets for one episode. Uses `last_val=0` when the episode ends (no bootstrap).
  - Two optimizers: `Adam(ac.pi.parameters(), lr=lr_pi)` and `Adam(ac.v.parameters(), lr=lr_vf)`.
  - Default hyperparameters follow Spinning Up VPG: `lr_pi=3e-4`, `lr_vf=1e-3`, `gamma=0.99`, `lam=0.95`, `train_v_iters=80`.

- **`core.MLPActorCritic`**: Unchanged; provides both policy (\(\pi\)) and value (\(v\)) used in `step()` and in the loss functions.

### Key Differences from Part 3 (No Baseline)

| Aspect | Part 3 (reward-to-go, no baseline) | Part 4 (GAE + value baseline) |
|--------|-------------------------------------|--------------------------------|
| Weight in policy gradient | Reward-to-go | GAE advantage \(\hat{A}\) |
| Value function | Not trained | Trained with MSE on returns |
| Optimizers | One (policy only) | Two (policy + value) |
| Advantage normalization | No | Yes (mean 0, std 1) |
| Value updates per epoch | 0 | `train_v_iters` (e.g. 80) |

## Testing

### CartPole-v1 (Discrete Actions)

**Purpose**: Check that GAE + value baseline works on a discrete action environment and compare to Part 3.

**How to run** (without baseline, Part 3):
```bash
python 3_continuous_actions_pg.py --env_name CartPole-v1 --epochs 50 --seed 0 --save_results
```

**How to run** (with GAE baseline, Part 4):
```bash
python 4_gae_baseline_pg.py --env_name CartPole-v1 --epochs 50 --seed 0 --save_results
```

**Results**: [Fill in after running: mean return per epoch, final performance, and any qualitative differences in learning speed or stability.]

**Observations**: [Discuss: Does the baseline version converge faster or more stably? Any sensitivity to lr_pi, lr_vf, or lam?]

### InvertedPendulum-v4 (Continuous Actions)

**Purpose**: Compare GAE + value baseline to reward-to-go without baseline on a continuous control task.

**Part 3 (no baseline)** – existing result file `results/continuous_pg_InvertedPendulum-v4_seed0.json` shows mean returns per epoch (e.g. final epochs often reach 1000, the max episode return).

**How to run** (with GAE baseline):
```bash
python 4_gae_baseline_pg.py --env_name InvertedPendulum-v4 --epochs 50 --seed 0 --save_results
```

**Results**: [Fill in: mean return per epoch for GAE baseline; compare learning curve and final performance to Part 3.]

**Observations**: [Discuss: Sample efficiency, stability, and final performance vs. Part 3.]

## Comparison: With vs. Without Baseline

- **Variance reduction**: The baseline (value function) subtracts an estimate of “how good the state is” from the return, so the policy gradient is weighted by advantage rather than raw return. This typically reduces variance and can speed up learning.

- **GAE vs. reward-to-go as weight**: GAE uses a \(\lambda\)-return style combination of TD errors, trading off bias and variance. With \(\lambda=0.95\) we get a good balance; with \(\lambda=1\) we recover the same expectation as reward-to-go but with a different (often more stable) estimate.

- **Stability**: Training the value function with multiple steps per epoch and normalizing advantages usually makes learning more stable than using raw reward-to-go without a baseline.

[After you run experiments, add a short summary: e.g. “On CartPole, GAE+baseline reached X return in Y epochs vs. Z for Part 3” and “On InvertedPendulum, …”.]

## Usage Instructions

### GAE + baseline (Part 4)
```bash
# CartPole
python 4_gae_baseline_pg.py --env_name CartPole-v1 --epochs 50 --save_results

# InvertedPendulum
python 4_gae_baseline_pg.py --env_name InvertedPendulum-v4 --epochs 50 --save_results

# Optional: tune GAE and learning rates
python 4_gae_baseline_pg.py --env_name InvertedPendulum-v4 --epochs 50 \
    --lr_pi 3e-4 --lr_vf 1e-3 --lam 0.95 --train_v_iters 80 --save_results
```

### Without baseline (Part 3, for comparison)
```bash
python 3_continuous_actions_pg.py --env_name CartPole-v1 --epochs 50 --save_results
python 3_continuous_actions_pg.py --env_name InvertedPendulum-v4 --epochs 50 --save_results
```

Results are written to `results/` as JSON (e.g. `gae_baseline_CartPole-v1_seed0.json`, `continuous_pg_InvertedPendulum-v4_seed0.json`).

## References

- Spinning Up VPG (GAE + value baseline): [vpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py) — advantage normalization (lines 79–81), value MSE loss, and `train_v_iters` value updates.
- GAE paper: *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (Schulman et al.).
