"""Bounded real-emulator validation, with optional external integrations.

Uses the current checkout's code/config; never edits the integration directory.
"""
import argparse
import gc
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--integration-dir")
    p.add_argument("--states", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    import custom_integrations
    if args.integration_dir:
        custom_integrations.INTEGRATIONS_DIR = os.path.abspath(args.integration_dir)
    custom_integrations.register()
    from rl.env import TrainingSpec
    from rl.afterstate import train_afterstate, unittest_afterstate, torch
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    spec = TrainingSpec(args.game, {"state": args.states})
    run = SimpleNamespace(game=args.game, save_dir=args.output, lr=None, gamma=None,
        batch_size=64, device=args.device, ent_coef=None, ent_coef_final=None,
        explore_steps=120000, timesteps=args.steps, resume=None,
        save_every=max(1, args.steps), unittest_samples=4)
    train_afterstate(run, spec, {"state": args.states})
    gc.collect()
    run.resume = os.path.join(args.output, "final.zip")
    unittest_afterstate(run, spec, {"state": spec.states[0]})


if __name__ == "__main__":
    main()
