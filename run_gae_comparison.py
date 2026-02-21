#!/usr/bin/env python3
"""
Run policy gradient with and without GAE baseline on CartPole and InvertedPendulum,
then print a short comparison. Requires: torch, gymnasium, and the project scripts.
"""
import json
import os
import sys

def run_cmd(cmd):
    import subprocess
    print(f"\n>>> {cmd}\n")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"Command failed with exit code {r.returncode}", file=sys.stderr)
        sys.exit(r.returncode)

def main():
    os.makedirs("results", exist_ok=True)
    seed = 0
    epochs = 50

    # Without baseline (Part 3)
    run_cmd(f"python 3_continuous_actions_pg.py --env_name CartPole-v1 --epochs {epochs} --seed {seed} --save_results")
    run_cmd(f"python 3_continuous_actions_pg.py --env_name InvertedPendulum-v4 --epochs {epochs} --seed {seed} --save_results")

    # With GAE baseline (Part 4)
    run_cmd(f"python 4_gae_baseline_pg.py --env_name CartPole-v1 --epochs {epochs} --seed {seed} --save_results")
    run_cmd(f"python 4_gae_baseline_pg.py --env_name InvertedPendulum-v4 --epochs {epochs} --seed {seed} --save_results")

    # Load and compare
    def load_returns(method, env, s):
        f = f"results/{method}_{env.replace('/', '_')}_seed{s}.json"
        if not os.path.isfile(f):
            return None
        with open(f) as fp:
            d = json.load(fp)
        return d.get("mean_returns", [])

    print("\n" + "="*60)
    print("COMPARISON (mean return per epoch, last 5 epochs)")
    print("="*60)
    for env in ["CartPole-v1", "InvertedPendulum-v4"]:
        key = env.replace("/", "_")
        no_base = load_returns("continuous_pg", env, seed)
        gae = load_returns("gae_baseline", env, seed)
        print(f"\n{env}:")
        if no_base:
            print(f"  Without baseline (Part 3): last 5 mean_returns = {no_base[-5:]}")
        else:
            print("  Without baseline: (no results file)")
        if gae:
            print(f"  GAE + baseline (Part 4):   last 5 mean_returns = {gae[-5:]}")
        else:
            print("  GAE + baseline: (no results file)")
    print()

if __name__ == "__main__":
    main()
