"""
seed_sweep.py
-------------
Run the same PINN architecture across multiple random seeds and report
which one converged best.

Each seed gets its own output directory (outputs_root/seed_<N>/), trains
independently via main.py as a subprocess, and --data_seed is held fixed
across all of them so every run trains on the identical dataset — the only
thing varying is weight initialisation and the training shuffle order.

Usage
-----
    python seed_sweep.py --seeds 1 2 3 4 5 --n_adam 20000 --n_lbfgs 500

Anything after the seed list is forwarded straight to main.py, so you can
pass --n_points, --mach_inf, --wedge_angle_deg, etc. the same way you would
normally.

After all runs finish, it prints a table of final total loss per seed
(read back from each run's training_state.pt) sorted best-to-worst, so you
can pick a winner or decide to ensemble the top few.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def build_parser():
    p = argparse.ArgumentParser(description="Multi-seed PINN sweep")
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="List of seeds to run, e.g. --seeds 1 2 3 4 5")
    p.add_argument("--data_seed", type=int, default=100,
                    help="Fixed dataset seed shared by every run in the "
                         "sweep, so only weight init / shuffling differs.")
    p.add_argument("--output_root", default="outputs_seed_sweep",
                    help="Each seed's run goes in <output_root>/seed_<N>/")
    p.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for subprocess runs.")
    return p


def main():
    args, extra = build_parser().parse_known_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        run_dir = root / f"seed_{seed}"
        print(f"\n{'='*60}")
        print(f"  Sweep: seed={seed}  data_seed={args.data_seed}  -> {run_dir}")
        print(f"{'='*60}")

        cmd = [
            args.python, "main.py",
            "--task", "train",
            "--output_dir", str(run_dir),
            "--seed", str(seed),
            "--data_seed", str(args.data_seed),
            *extra,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[Sweep] seed={seed} FAILED (exit code {ret.returncode}); skipping.")
            continue

        state_path = run_dir / "checkpoints" / "training_state.pt"
        if not state_path.exists():
            print(f"[Sweep] seed={seed}: no training_state.pt found, skipping summary.")
            continue

        ckpt = torch.load(state_path, map_location="cpu")
        history = ckpt.get("history", {})
        final_loss = history["total"][-1] if history.get("total") else float("nan")
        final_step = history["step"][-1] if history.get("step") else 0
        results.append((seed, final_step, final_loss, str(run_dir)))

    if not results:
        print("\n[Sweep] No successful runs to summarise.")
        return

    results.sort(key=lambda r: r[2])  # sort by final loss, best first

    print(f"\n{'='*60}")
    print(f"  Sweep results (best first)")
    print(f"{'='*60}")
    print(f"  {'seed':>6}  {'step':>8}  {'final_loss':>14}  output_dir")
    for seed, step, loss, run_dir in results:
        print(f"  {seed:>6}  {step:>8}  {loss:>14.6e}  {run_dir}")

    best = results[0]
    print(f"\n[Sweep] Best: seed={best[0]}  final_loss={best[2]:.6e}  -> {best[3]}/model_final.pt")


if __name__ == "__main__":
    main()
