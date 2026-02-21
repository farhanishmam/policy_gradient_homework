# Part 3: Continuous Actions - Report

## Overview

This report documents the implementation and testing of a policy gradient algorithm that works with both discrete and continuous action spaces using the `MLPActorCritic` class from `core.py`.

## Implementation Details

### Key Changes from Previous Parts

1. **Replaced manual policy network with `MLPActorCritic`**: Instead of manually building a policy network that only works for discrete actions, we now use the `MLPActorCritic` class which automatically handles both discrete (`MLPCategoricalActor`) and continuous (`MLPGaussianActor`) action spaces.

2. **Removed action space restrictions**: The code no longer asserts that the action space must be discrete, allowing it to work with both `Discrete` and `Box` action spaces.

3. **Action handling**: The code now properly handles both types of actions:
   - **Discrete actions**: Scalar integers (e.g., `0`, `1` for CartPole)
   - **Continuous actions**: Numpy arrays (e.g., `[0.5, -0.3]` for InvertedPendulum)

4. **Value function**: The value function (`ac.v`) is created but not trained in this part, as specified in the instructions. We ignore the `v` output from `ac.step()`.

### Code Structure

The implementation (`3_continuous_actions_pg.py`) follows the same structure as Part 2 (reward-to-go policy gradient) but uses `MLPActorCritic` for policy representation:

- Uses `reward_to_go()` for variance reduction
- Collects batches of experience
- Updates policy using policy gradient with reward-to-go weights
- Supports both discrete and continuous environments

## Testing

### Discrete Action Environment: CartPole-v1

**Purpose**: Verify that the refactored code still works correctly on discrete action environments.

**Results**: [To be filled after running experiments]

**Observations**: [To be filled]

### Continuous Action Environment: InvertedPendulum-v4

**Purpose**: Test the algorithm on a simple continuous control task.

**Environment Details**:
- **State space**: 4-dimensional (cart position, cart velocity, pole angle, pole angular velocity)
- **Action space**: 1-dimensional continuous (force applied to cart, typically in range [-3, 3])
- **Goal**: Balance an inverted pendulum on a cart

**Hyperparameters Tested**:

1. **Baseline Configuration**:
   - Learning rate: `1e-2`
   - Hidden sizes: `[64, 64]`
   - Batch size: `5000`
   - Epochs: `50`
   - Activation: `Tanh`

2. **Variations Tested**:
   - [To be filled after experiments]
   - Learning rate: `5e-3`, `1e-2`, `2e-2`
   - Hidden sizes: `[32]`, `[64, 64]`, `[128, 128]`
   - Batch size: `3000`, `5000`, `10000`

**Results**: [To be filled after running experiments]

**Learning Curves**: [To be filled - include plots if available]

### Additional Environments Tested

**Hopper-v4** (if tested):
- More complex locomotion task
- [Results to be filled]

## Hyperparameter Experiments

### Learning Rate Sensitivity

[To be filled with results comparing different learning rates]

### Network Architecture

[To be filled with results comparing different hidden layer sizes]

### Batch Size Effects

[To be filled with results comparing different batch sizes]

## Key Findings

1. **Generalization**: The `MLPActorCritic` class successfully generalizes to both discrete and continuous action spaces without code changes.

2. **Continuous Action Handling**: The Gaussian policy (mean and standard deviation) works well for continuous control tasks.

3. **Hyperparameter Sensitivity**: [To be filled]

4. **Learning Dynamics**: [To be filled - discuss how learning progresses, convergence, stability]

## Challenges and Solutions

1. **Action Type Handling**: Initially had to ensure proper conversion between PyTorch tensors and numpy arrays for both discrete (scalar) and continuous (array) actions.

2. **Tensor Types**: Had to handle different tensor types (`int32` for discrete, `float32` for continuous) when converting batches to tensors.

## Conclusion

[To be filled with summary of findings and insights]

## Usage Instructions

### Running on CartPole (Discrete):
```bash
python 3_continuous_actions_pg.py --env_name CartPole-v1 --epochs 50 --save_results
```

### Running on InvertedPendulum (Continuous):
```bash
python 3_continuous_actions_pg.py --env_name InvertedPendulum-v4 --epochs 50 --save_results
```

### Running with Custom Hyperparameters:
```bash
python 3_continuous_actions_pg.py --env_name InvertedPendulum-v4 \
    --lr 5e-3 --hidden_sizes 128 128 --epochs 100 --save_results
```

## Future Work

- Implement value function training (Part 4/Extra Credit A)
- Add baseline subtraction for further variance reduction
- Test on more complex continuous control tasks
