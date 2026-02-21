# Part 1a: Basic Policy Gradient Algorithm - Results and Analysis

## Overview

This report analyzes the results from running the simple policy gradient algorithm (`1_simple_pg_gymnasium.py`) on the CartPole-v1 environment for 50 epochs. The algorithm uses the simplest formulation of policy gradient, where the gradient is computed using the full episode return as the weight for each action's log probability.

## Results Summary

The training run shows clear evidence of learning, with episode returns increasing from approximately **20.6** at epoch 0 to approximately **391.9** at epoch 49—an improvement of roughly **19x**.

### Key Metrics Over Training

- **Initial Performance (Epochs 0-5)**: Returns range from 20.6 to 36.4, showing rapid initial improvement
- **Mid Training (Epochs 10-25)**: Returns increase from 55.5 to 194.3, demonstrating continued learning
- **Late Training (Epochs 30-49)**: Returns stabilize around 300-400, with some variance

## Is the Agent Learning?

**Yes, the agent is clearly learning.** The evidence for this includes:

1. **Dramatic Performance Increase**: Episode returns increase from ~20.6 to ~391.9 over 50 epochs, representing nearly a 20-fold improvement.

2. **Consistent Upward Trend**: Despite variance, there is a clear overall upward trajectory in performance. The agent starts with very short episodes (around 20 steps) and learns to maintain the pole for much longer (approaching 400 steps).

3. **Loss Behavior**: The loss t increases from 17.9 to 227.1 over training. This is expected behavior for this policy gradient formulation, as the loss is computed as `-(log_prob * return).mean()`. As returns increase, the magnitude of the loss increases accordingly, even though the policy is improving.

## What Happens to Performance Over Time?

Performance shows a **non-monotonic but generally increasing trend**:

1. **Rapid Initial Learning (Epochs 0-10)**: Returns increase from 20.6 to 55.5, showing the fastest relative improvement.

2. **Steady Improvement (Epochs 10-30)**: Returns continue to increase, reaching 255.9 by epoch 30, though the rate of improvement slows.

3. **High Performance with Variance (Epochs 30-49)**: Returns stabilize in the 300-400 range, but with noticeable variance:
   - Epoch 36: 378.6
   - Epoch 37: 352.5 (decrease)
   - Epoch 38: 387.2 (recovery)
   - Epoch 41: 440.2 (peak)
   - Epoch 42: 362.9 (decrease)
   - Epoch 49: 391.9

## Is Performance Monotonically Improving?

**No, performance is not monotonically improving.** There are several instances where returns decrease between consecutive epochs:

- Epoch 6 → 7: 42.7 → 40.3
- Epoch 11 → 12: 59.8 → 58.6
- Epoch 13 → 14: 66.9 → 66.0
- Epoch 20 → 21: 97.5 → 94.9
- Epoch 27 → 28: 220.8 → 219.5
- Epoch 30 → 31: 255.9 → 240.0
- Epoch 31 → 32: 240.0 → 238.2
- Epoch 36 → 37: 378.6 → 352.5 (significant decrease)
- Epoch 40 → 41: 381.5 → 440.2 (increase)
- Epoch 41 → 42: 440.2 → 362.9 (significant decrease)
- Epoch 43 → 44: 381.5 → 405.2 (increase)
- Epoch 44 → 45: 405.2 → 393.2 (decrease)

This non-monotonic behavior is particularly noticeable in later epochs, where the variance increases.

## Does This Make Sense?

**Yes, this behavior is expected and makes sense** for several reasons:

1. **Stochastic Nature of Policy Gradients**: Policy gradient methods are inherently stochastic. The policy samples actions probabilistically, and each epoch collects a batch of episodes that may have different outcomes due to randomness in:
   - Action sampling from the policy
   - Environment dynamics
   - Initial states

2. **High Variance in Gradient Estimates**: The simple policy gradient formulation used here (full episode return as weight) has high variance. This is because:
   - All actions in an episode receive the same weight (the full episode return)
   - Good and bad actions within the same episode are weighted equally
   - This leads to noisy gradient estimates, which can cause performance fluctuations

3. **Exploration vs. Exploitation Trade-off**: As the policy improves, it may occasionally explore suboptimal actions or encounter unfavorable initial states, leading to temporary performance dips.

4. **CartPole Environment Characteristics**: CartPole-v1 has a maximum episode length of 500 steps. The agent reaching returns in the 300-400 range suggests it's performing well but hasn't yet mastered the task completely, leaving room for variance.

5. **Loss Interpretation**: The increasing loss value is not a sign of poor learning. In this formulation, loss = `-(log_prob * return).mean()`. As returns increase, the loss magnitude increases, but the gradient updates still improve the policy by increasing the probability of high-return trajectories.

## Conclusion

The simple policy gradient algorithm successfully learns to solve the CartPole task, demonstrating clear learning over 50 epochs. While performance is not monotonically improving due to the stochastic nature of the algorithm and high variance in gradient estimates, the overall trend is strongly positive. The non-monotonic behavior is expected and reflects the inherent variance in policy gradient methods, particularly when using the simple formulation with full episode returns as weights.
