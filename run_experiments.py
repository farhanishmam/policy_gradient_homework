"""
Run experiments comparing simple policy gradient vs reward-to-go policy gradient.
Runs each method multiple times with different seeds, then plots average return vs epoch.

Usage:
    python run_experiments.py                       # default: 5 seeds, 50 epochs
    python run_experiments.py --epochs 10 --num_seeds 2  # quick test
"""
import subprocess
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def run_single(script, seed, epochs, env_name):
    """Run a single training script with given seed and epochs."""
    cmd = [
        'python', script,
        '--env_name', env_name,
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--save_results'
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
    return result.returncode


def load_results(pattern, num_seeds, results_dir='results'):
    """Load results from JSON files matching the pattern."""
    all_returns = []
    for seed in range(num_seeds):
        fname = os.path.join(results_dir, f'{pattern}_seed{seed}.json')
        with open(fname, 'r') as f:
            data = json.load(f)
        all_returns.append(data['mean_returns'])
    return np.array(all_returns)


def plot_comparison(simple_returns, rtg_returns, save_path='comparison_plot.png'):
    """Plot average return vs epoch for both methods with std shading."""
    epochs = np.arange(simple_returns.shape[1])

    simple_mean = np.mean(simple_returns, axis=0)
    simple_std = np.std(simple_returns, axis=0)

    rtg_mean = np.mean(rtg_returns, axis=0)
    rtg_std = np.std(rtg_returns, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, simple_mean, label='Simple PG', color='tab:blue')
    plt.fill_between(epochs, simple_mean - simple_std, simple_mean + simple_std,
                     alpha=0.2, color='tab:blue')

    plt.plot(epochs, rtg_mean, label='Reward-to-Go PG', color='tab:orange')
    plt.fill_between(epochs, rtg_mean - rtg_std, rtg_mean + rtg_std,
                     alpha=0.2, color='tab:orange')

    plt.xlabel('Epoch')
    plt.ylabel('Average Return')
    plt.title('Simple PG vs Reward-to-Go PG on CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'\nPlot saved to {save_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run PG experiments and plot comparison.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Gymnasium env name')
    parser.add_argument('--skip_run', action='store_true', help='Skip running, just plot from existing results')
    args = parser.parse_args()

    if not args.skip_run:
        # Run simple PG experiments
        print("=" * 60)
        print("Running Simple Policy Gradient experiments")
        print("=" * 60)
        for seed in range(args.num_seeds):
            run_single('1_simple_pg_gymnasium.py', seed, args.epochs, args.env_name)

        # Run reward-to-go PG experiments
        print("=" * 60)
        print("Running Reward-to-Go Policy Gradient experiments")
        print("=" * 60)
        for seed in range(args.num_seeds):
            run_single('2_reward_to_go_pg.py', seed, args.epochs, args.env_name)

    # Load results and plot
    print("\nLoading results and generating plot...")
    simple_returns = load_results('simple_pg', args.num_seeds)
    rtg_returns = load_results('reward_to_go', args.num_seeds)
    plot_comparison(simple_returns, rtg_returns)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary (final epoch):")
    print(f"  Simple PG:       mean={np.mean(simple_returns[:, -1]):.1f}, std={np.std(simple_returns[:, -1]):.1f}")
    print(f"  Reward-to-Go PG: mean={np.mean(rtg_returns[:, -1]):.1f}, std={np.std(rtg_returns[:, -1]):.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
