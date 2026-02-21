import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import json
import os
from core import MLPActorCritic


def reward_to_go(rews):
    """
    Compute reward-to-go for each timestep.
    Given rewards [r0, r1, ..., rT], returns [r0+r1+...+rT, r1+...+rT, ..., rT].
    """
    n = len(rews)
    rtgs = np.zeros_like(rews, dtype=np.float32)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def train(env_name='CartPole-v1', hidden_sizes=[64, 64], lr=1e-2,
          epochs=50, batch_size=5000, render=False, seed=0, save_results=False):

    # set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # make environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    # Check that observation space is Box (vector-based)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    
    # Use MLPActorCritic which handles both discrete and continuous actions
    ac = MLPActorCritic(env.observation_space, env.action_space, 
                        hidden_sizes=hidden_sizes, activation=nn.Tanh)

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        # Get policy distribution and log probability
        pi, logp_a = ac.pi(obs, act)
        return -(logp_a * weights).mean()

    # make optimizer (only optimize policy, not value function for now)
    optimizer = Adam(ac.pi.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, info = env.reset()
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment using MLPActorCritic
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, v, logp = ac.step(obs_tensor)
            
            # Handle both discrete (scalar) and continuous (array) actions
            if isinstance(env.action_space, Discrete):
                act = int(a)  # discrete action is a scalar
            else:
                act = a  # continuous action is already an array
            
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is the reward-to-go from that step
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, info = env.reset()
                done, ep_rews = False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # Convert actions to tensor - handle both discrete and continuous
        obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        if isinstance(env.action_space, Discrete):
            act_tensor = torch.as_tensor(batch_acts, dtype=torch.int32)
        else:
            act_tensor = torch.as_tensor(batch_acts, dtype=torch.float32)
        weights_tensor = torch.as_tensor(batch_weights, dtype=torch.float32)

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs_tensor, act_tensor, weights_tensor)
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    all_mean_rets = []  # for saving results
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        mean_ret = np.mean(batch_rets)
        all_mean_rets.append(float(mean_ret))
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss.item(), mean_ret, np.mean(batch_lens)))

        # Part 1b: visually render 1 episode after each training epoch
        if render:
            render_env = gym.make(env_name, render_mode='human')
            obs_render, _ = render_env.reset()
            done_render = False
            while not done_render:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_render, dtype=torch.float32)
                    a_render, _, _ = ac.step(obs_tensor)
                    if isinstance(env.action_space, Discrete):
                        act_render = int(a_render)
                    else:
                        act_render = a_render
                obs_render, _, terminated, truncated, _ = render_env.step(act_render)
                done_render = terminated or truncated
            render_env.close()

    # save results to JSON if requested
    if save_results:
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        fname = os.path.join(results_dir, f'continuous_pg_{env_name.replace("/", "_")}_seed{seed}.json')
        with open(fname, 'w') as f:
            json.dump({'env': env_name, 'method': 'continuous_pg', 'seed': seed,
                       'epochs': epochs, 'mean_returns': all_mean_rets}, f)
        print(f'Results saved to {fname}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer sizes for the neural network')
    args = parser.parse_args()
    print(f'\nUsing MLPActorCritic for policy gradient (works with discrete and continuous actions).')
    print(f'Environment: {args.env_name}')
    print(f'Action space: {gym.make(args.env_name).action_space}\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr,
          epochs=args.epochs, seed=args.seed, save_results=args.save_results,
          hidden_sizes=args.hidden_sizes)
