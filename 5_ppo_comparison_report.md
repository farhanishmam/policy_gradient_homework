# Extra Credit B: PPO vs. Simple Policy Gradient — Report

This report compares the **Spinning Up PPO implementation** ([documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html), [ppo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)) with our **simple policy gradient code** (Part 3: `3_continuous_actions_pg.py` with reward-to-go; Part 4: `4_gae_baseline_pg.py` with GAE and value baseline). The goal is to highlight what is the same, what is different, what was learned, and what remains unclear.

---

## What Is the Same

- **Actor–critic interface**: PPO uses the same `MLPActorCritic` from `core.py`: a module with `step(obs)`, `act(obs)`, and submodules `pi` (policy) and `v` (value). So the *architecture* (policy net + value net, discrete vs continuous via `Box`/`Discrete`) is shared.

- **On-policy data**: Both collect a batch of trajectories with the *current* policy, then update. No replay buffer; data is thrown away after each update.

- **GAE and value targets**: Like our Part 4 (GAE baseline), PPO uses **GAE-Lambda** for advantages and **reward-to-go** (discounted returns) as targets for the value function. The buffer’s `finish_path(last_val)` does the same thing: `deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]`, `adv_buf = discount_cumsum(deltas, gamma*lam)`, `ret_buf = discount_cumsum(rews, gamma)[:-1]`.

- **Advantage normalization**: Both normalize advantages to mean 0 and std 1 before using them in the policy update (Spinning Up lines 89–91; our `4_gae_baseline_pg.py` does the same).

- **Value function training**: Same idea: MSE between \(V(s)\) and the return. Same pattern of **multiple value gradient steps per epoch** (`train_v_iters=80` in both).

- **Two optimizers**: One for the policy, one for the value function, with separate learning rates (`pi_lr`, `vf_lr`).

- **Core utilities**: PPO uses the same `core.discount_cumsum`, `core.combined_shape`, and the same policy/value interface we rely on.

So in terms of *data collection*, *advantage computation*, and *value learning*, PPO and our GAE-baseline VPG are very similar. The big difference is *how* the policy is updated.

---

## What Is Different

### 1. Policy objective: clipping instead of plain policy gradient

- **Our code (VPG)**: Policy loss is  
  `loss_pi = -(logp_a * adv).mean()`  
  i.e. maximize \(\mathbb{E}[\log \pi(a|s)\, A]\). One gradient step per epoch; no limit on how much the policy can change.

- **PPO**: Policy loss is the **clipped surrogate objective**  
  `ratio = exp(logp - logp_old)`  
  `loss_pi = -(min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)).mean()`  
  So we maximize \(\mathbb{E}[\min(r_t(\theta)\,\hat{A}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A})]\), where \(r_t(\theta) = \pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)\). This **limits how much the new policy can deviate from the old**: when the ratio goes outside \([1-\epsilon, 1+\epsilon]\), the objective no longer improves by moving further. That’s the “proximal” idea: keep updates conservative to avoid performance collapse.

### 2. Use of *old* log probabilities

- **Our code**: We only use the *current* policy’s \(\log \pi(a|s)\) in the loss. We don’t need to store or use old log probs.

- **PPO**: The buffer stores **log probabilities under the policy that collected the data** (`logp_buf`). During the update, `logp_old` is read from the buffer and used to form the ratio \(\pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s) = \exp(\log p - \log p_{\text{old}})\). So PPO is explicitly a **relative** update (new vs old policy), while our VPG is an absolute policy gradient step.

### 3. Multiple policy gradient steps per batch

- **Our code**: Exactly **one** policy gradient step per epoch (one batch of data).

- **PPO**: Up to **`train_pi_iters`** (e.g. 80) gradient steps on the *same* batch. So we **reuse** the same data for many policy updates. That’s only valid because the objective is a *surrogate* that’s tied to the old policy; once the policy drifts too far, the surrogate is no longer trustworthy, which leads to the next point.

### 4. Early stopping by KL

- **Our code**: No KL term; we always take our single step.

- **PPO**: After each policy gradient step it computes an approximate KL between new and old policy (e.g. `approx_kl = (logp_old - logp).mean().item()`). If `kl > 1.5 * target_kl`, it **stops** taking more policy steps for that epoch. So the “multiple steps” are capped when the policy has changed “too much,” avoiding the worst case of reusing stale data.

### 5. Buffering and epoch structure

- **Our code**: We collect until we have more than `batch_size` *steps* (e.g. 5000), with episodes possibly finishing at different lengths. We don’t pre-allocate; we use lists and break when done.

- **PPO**: Fixed **`steps_per_epoch`** (e.g. 4000). A pre-allocated buffer is filled exactly that many steps; trajectories can be cut mid-episode, in which case `last_val = V(s_T)` is used for bootstrapping in `finish_path(last_val)`. So PPO has a fixed batch size and explicit handling of truncated trajectories.

### 6. MPI / parallelization

- **Our code**: Single process; one environment.

- **PPO**: Uses **MPI** to run multiple workers (e.g. 4 CPUs). Each worker collects a fraction of `steps_per_epoch`, then gradients and statistics are averaged across processes (`mpi_avg_grads`, `mpi_statistics_scalar`). So the same batch size in “steps” is spread over multiple envs for faster data collection.

### 7. Logging and saving

- **Our code**: Simple prints and optional JSON for mean returns.

- **PPO**: EpochLogger, TensorBoard-style logging, saving the full actor-critic, and extra stats (KL, clip fraction, stop iteration, etc.) for debugging and tuning.

---

## What I Learned

1. **Clipping as a trust region**: The PPO clip objective is a first-order way to get a “trust region” effect without solving a constrained optimization like TRPO. The min/max and the ratio make it clear: the objective stops rewarding the policy for moving further once \(\pi_\theta(a|s)\) is too far from \(\pi_{\theta_{\text{old}}}(a|s)\).

2. **Data reuse**: Reusing the same batch for many gradient steps is what makes PPO sample-efficient relative to VPG, but it only works because (a) the objective is a surrogate tied to the old policy, and (b) early stopping by KL prevents the policy from changing so much that the surrogate becomes invalid.

3. **Old log probs are essential**: Storing and using `logp_old` is what defines “old” vs “new” policy and makes the ratio and clipping meaningful. In VPG we don’t need this because we don’t form a ratio.

4. **Same GAE/value machinery**: Seeing the same GAE and value-target logic in PPO as in our Part 4 made it clear that PPO is “VPG + clipped objective + multiple policy steps + KL early stopping,” not a different way to compute advantages or value targets.

5. **Implementation details**: Bootstrapping with `last_val` when a trajectory is cut by the epoch, pre-allocated buffers with `combined_shape`, and the exact clipping logic (e.g. `clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv` then `min(ratio*adv, clip_adv)`) were useful to see in one place.

---

## What Is Still Confusing or Unclear

1. **Choice of `target_kl` and `1.5 * target_kl`**: The docstring says “roughly what KL we think is appropriate.” Why 1.5× that for early stopping? Is it purely empirical? A short note or reference on how to set `target_kl` in practice would help.

2. **Clip fraction interpretation**: They log “ClipFrac” (fraction of samples where the ratio was outside the clip range). High clip fraction means we’re hitting the ceiling often—but is that “good” (we’re limiting change) or “bad” (we might be limiting progress)? The link between clip fraction and learning speed/stability is not obvious from the code alone.

3. **No gradient clipping**: Some PPO implementations clip gradients for the policy or value. This one doesn’t. It would be useful to know whether that was a deliberate choice and when gradient clipping might be added.

4. **Interaction of `train_pi_iters` and early stopping**: If we almost always stop after a few iterations because of KL, then `train_pi_iters=80` is effectively a cap rather than the typical number of steps. How to set `train_pi_iters` and `target_kl` together (e.g. for different envs or batch sizes) is not clearly documented.

5. **MPI dependency**: Running the script as-is requires OpenMPI and the Spinning Up env setup. The logic (buffer, GAE, clip loss, early stopping) is clear, but to run it without MPI you’d need to strip out `mpi_fork`, `sync_params`, `mpi_avg_grads`, `mpi_statistics_scalar`, and replace `mpi_statistics_scalar` with plain numpy mean/std for advantage normalization. The documentation mentions this can be non-trivial on non-Ubuntu systems; a “single-process” mode or a note in the code would make experimentation easier.

---

## Summary Table

| Aspect | Our VPG (Part 3 / Part 4) | Spinning Up PPO |
|--------|----------------------------|------------------|
| Policy loss | \(-\mathbb{E}[\log \pi(a|s)\, A]\) | Clipped surrogate \(\min(r_t A, \text{clip}(r_t)A\) |
| Old log probs | Not used | Stored in buffer, used in ratio |
| Policy steps per epoch | 1 | Up to `train_pi_iters` (e.g. 80), with KL early stop |
| Value loss | MSE(\(V\), return) | Same |
| Value steps per epoch | `train_v_iters` (e.g. 80) | Same |
| Advantage | GAE (Part 4) or reward-to-go (Part 3) | GAE only |
| Advantage normalization | Yes (Part 4) | Yes |
| Batch size | Variable (e.g. > 5000 steps) | Fixed `steps_per_epoch` (e.g. 4000) |
| Trajectory cutoff | Only at episode end | Can cut mid-episode; bootstrap with \(V(s_T)\) |
| Parallelization | None | MPI multi-worker |
| Trust region / stability | None (VPG) or baseline only (Part 4) | Clip + KL early stopping |

---

## Optional: Running PPO Without MPI

To run the Spinning Up PPO logic without OpenMPI:

1. Remove or bypass `mpi_fork(args.cpu)` so only one process runs.
2. Replace `mpi_statistics_scalar(self.adv_buf)` with `np.mean(self.adv_buf)`, `np.std(self.adv_buf)` (and handle the case when std is 0).
3. Remove or no-op `sync_params(ac)` and `mpi_avg_grads(ac.pi)` / `mpi_avg_grads(ac.v)`.
4. Replace `mpi_avg(pi_info['kl'])` with `pi_info['kl']` for the early-stopping check.
5. Adjust imports and `obs_dim`/`act_dim` (Spinning Up uses `env.observation_space.shape` which is a tuple; may need to flatten for `combined_shape`).
6. Replace `gym` with `gymnasium` and update `env.reset()` and `env.step()` return signatures (e.g. `obs, info = env.reset()`, `terminated, truncated`).

An LLM or a small refactor script can do this so you can run PPO locally and compare learning curves to your VPG/GAE baseline on the same envs and seeds.
