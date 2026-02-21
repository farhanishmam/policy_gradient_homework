"""
Policy gradient with GAE (Generalized Advantage Estimation) and a learned value function baseline.
Uses MLPActorCritic; policy updated with advantage weights, value function trained with MSE on returns.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import json
import os
from core import MLPActorCritic, discount_cumsum


def gae_and_returns(rewards, values, last_val, gamma=0.99, lam=0.95):
    """
    Compute GAE-Lambda advantages and reward-to-go (returns) for value targets.
    rewards: list of rewards [r_0, ..., r_{T-1}] for the episode
    values: list of V(s_t) [v_0, ..., v_{T-1}]
    last_val: bootstrap value for V(s_T); use 0 if episode ended naturally.
    Returns:
        adv: advantages (same length as rewards)
        ret: returns = discounted sum of future rewards, used as value targets
    """
    rews = np.append(np.array(rewards, dtype=np.float32), last_val)
    vals = np.append(np.array(values, dtype=np.float32), last_val)
    # TD errors: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    # GAE: A_t = sum_l (gamma*lambda)^l * delta_{t+l}
    adv = discount_cumsum(deltas, gamma * lam)
    # Returns (reward-to-go) for value function targets
    ret = discount_cumsum(rews, gamma)[:-1]
    return adv.astype(np.float32), ret.astype(np.float32)


def train(env_name='CartPole-v1', hidden_sizes=(64, 64), lr_pi=3e-4, lr_vf=1e-3,
          epochs=50, batch_size=5000, gamma=0.99, lam=0.95, train_v_iters=80,
          render=False, seed=0, save_results=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.reset(seed=seed)

    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."

    ac = MLPActorCritic(env.observation_space, env.action_space,
                        hidden_sizes=hidden_sizes, activation=nn.Tanh)

    def compute_loss_pi(obs, act, adv):
        pi, logp_a = ac.pi(obs, act)
        return -(logp_a * adv).mean()

    def compute_loss_v(obs, ret):
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=lr_pi)
    vf_optimizer = Adam(ac.v.parameters(), lr=lr_vf)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_adv = []
        batch_ret = []
        batch_rets = []
        batch_lens = []

        obs, info = env.reset()
        done = False
        ep_rews = []
        ep_vals = []
        ep_logps = []
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, v, logp = ac.step(obs_tensor)

            if isinstance(env.action_space, Discrete):
                act = int(a)
            else:
                act = a

            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)
            ep_vals.append(float(v))
            ep_logps.append(float(logp))

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # last_val = 0 (episode ended; no bootstrap)
                last_val = 0.0
                adv, ret = gae_and_returns(ep_rews, ep_vals, last_val, gamma=gamma, lam=lam)
                batch_adv += list(adv)
                batch_ret += list(ret)

                obs, info = env.reset()
                done, ep_rews, ep_vals, ep_logps = False, [], [], []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        # Convert to tensors
        obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        if isinstance(env.action_space, Discrete):
            act_tensor = torch.as_tensor(batch_acts, dtype=torch.int32)
        else:
            act_tensor = torch.as_tensor(batch_acts, dtype=torch.float32)
        adv_tensor = torch.as_tensor(batch_adv, dtype=torch.float32)
        ret_tensor = torch.as_tensor(batch_ret, dtype=torch.float32)

        # Normalize advantages (recommended for stability)
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std()
        if adv_std > 1e-8:
            adv_tensor = (adv_tensor - adv_mean) / adv_std

        # Policy gradient step
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(obs_tensor, act_tensor, adv_tensor)
        loss_pi.backward()
        pi_optimizer.step()

        # Value function: multiple gradient steps for stability
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(obs_tensor, ret_tensor)
            loss_v.backward()
            vf_optimizer.step()

        return loss_pi.item(), loss_v.item(), batch_rets, batch_lens

    all_mean_rets = []
    for i in range(epochs):
        loss_pi, loss_v, batch_rets, batch_lens = train_one_epoch()
        mean_ret = np.mean(batch_rets)
        all_mean_rets.append(float(mean_ret))
        print('epoch: %3d \t loss_pi: %.3f \t loss_v: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, loss_pi, loss_v, mean_ret, np.mean(batch_lens)))

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

    if save_results:
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        fname = os.path.join(results_dir, f'gae_baseline_{env_name.replace("/", "_")}_seed{seed}.json')
        with open(fname, 'w') as f:
            json.dump({'env': env_name, 'method': 'gae_baseline', 'seed': seed,
                       'epochs': epochs, 'mean_returns': all_mean_rets}, f)
        print(f'Results saved to {fname}')

    return all_mean_rets


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr_pi', type=float, default=3e-4, help='Policy learning rate')
    parser.add_argument('--lr_vf', type=float, default=1e-3, help='Value function learning rate')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--train_v_iters', type=int, default=80,
                        help='Value function gradient steps per epoch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 64])
    args = parser.parse_args()
    print('\nPolicy gradient with GAE and learned value function baseline.')
    print(f'Environment: {args.env_name}')
    print(f'Action space: {gym.make(args.env_name).action_space}\n')
    train(env_name=args.env_name, hidden_sizes=tuple(args.hidden_sizes),
          lr_pi=args.lr_pi, lr_vf=args.lr_vf, epochs=args.epochs,
          batch_size=args.batch_size, gamma=args.gamma, lam=args.lam,
          train_v_iters=args.train_v_iters, render=args.render, seed=args.seed,
          save_results=args.save_results)
