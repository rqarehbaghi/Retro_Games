#!/usr/bin/env python3
"""Play one checkpoint over the frozen suite, or one (checkpoint, state) pair.

    # one pair, in this process (what the sweep launches per worker)
    python tools/progression/evaluate.py --one --checkpoint <zip|control> --state prog_01

    # a timing probe before committing to the full sweep
    python tools/progression/evaluate.py --probe 10

    # the control, then everything
    python tools/progression/evaluate.py --checkpoints control
    python tools/progression/evaluate.py --checkpoints all

Design notes that matter:

- ONE PROCESS PER PAIR. stable-retro allows a single emulator instance per
  process, and a crash in one run must not take the sweep with it. Workers are
  subprocesses; this module is both the launcher and the worker.
- ATOMIC CACHE. A result is written to a temporary file and renamed, so an
  interrupted run can never leave a half-written entry that looks complete.
  The key covers everything that can change an outcome (see ids.run_key).
- EQUAL OPPORTUNITY is a PLACEMENT cap, not a time cap: higher levels drop
  faster and animations eat frames, so a time budget would hand different
  checkpoints different numbers of moves.
- `game_over` and `placement_cap` are never merged. A capped trajectory is
  censored, not finished, and reporting them together would overstate exactly
  the checkpoints that reach the cap.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import ids                                # noqa: E402

OUT = os.path.join(ROOT, "progression_out")
CACHE = os.path.join(OUT, "cache")
CONTROL = "control"
MIN_FPS = 100.0          # a sane floor; the point is to catch accidental throttling


def manifest():
    with open(os.path.join(OUT, "suite_manifest.json")) as fh:
        return json.load(fh)


def checkpoint_list(folder):
    """Numbered checkpoints, ascending, with `final.zip` dropped when it is a
    byte-for-byte duplicate of the last one (it always has been so far)."""
    entries = []
    for name in os.listdir(folder):
        if name.startswith("ckpt_") and name.endswith("_steps.zip"):
            entries.append((int(name.split("_")[1]), os.path.join(folder, name)))
    entries.sort()
    out = [{"name": "%dk" % (step // 1000), "steps": step, "path": path,
            "sha256": ids.sha256_file(path)} for step, path in entries]
    final = os.path.join(folder, "final.zip")
    if os.path.exists(final):
        digest = ids.sha256_file(final)
        if not any(c["sha256"] == digest for c in out):
            out.append({"name": "final", "steps": None, "path": final, "sha256": digest})
    return out


# ----------------------------------------------------------------- worker --
def run_one(checkpoint, state, cap, player, deterministic=True):
    """Play one trajectory. Returns the result dict written to the cache."""
    import numpy as np                                            # noqa: F401
    import custom_integrations                                    # noqa: F401
    from rl.env import make_env, TrainingSpec
    from rl.vars import GameVars
    from rl.simulators import get_simulator
    from rl.afterstate import AfterstateAgent

    overrides = {"player": player, "state": state}
    spec = TrainingSpec(manifest()["game"], overrides)
    cfg = spec.afterstate_config
    sim = get_simulator(cfg.get("simulator") or spec.features_name or spec.game, config=cfg)
    agent = AfterstateAgent(input_dim=sim.feature_dim, device="cpu",
                            model_type=cfg.get("model_type", "mlp"),
                            zero_init_value=(checkpoint == CONTROL))
    if checkpoint != CONTROL:
        agent.load(checkpoint)
    gv = GameVars(spec.game, entry=spec.entry)
    lines_var = cfg.get("lines_var", "lines_p{player}").replace("{player}", str(player))

    env = make_env(spec.game, overrides)
    obs, info = env.reset()
    lines0 = int(info.get(lines_var, 0) or 0)
    placements, pending, frames = 0, False, 0
    end = "placement_cap"
    started = time.time()
    while placements < cap:
        action = None
        if not pending:
            cands = sim.get_candidates(obs=obs, ram=getattr(env.unwrapped, "ram", None),
                                       info=info, vars=gv, spec=spec)
            if cands:
                action = agent.select_action(cands, epsilon=0.0)[0]
                pending, placements = True, placements + 1
        obs, _r, term, trunc, info = env.step(action)
        frames += 1
        if info.get("afterstate_ready", True):
            pending = False
        if term:
            end = "game_over"
            break
        if trunc:
            end = "truncated"
            break
    elapsed = max(1e-6, time.time() - started)
    # Emulator FRAMES, not macro placements: one placement is dozens of frames,
    # and the uncapped-execution check has to look at the real rate.
    emu_frames = int(getattr(env.unwrapped, "frames", 0) or 0)
    env.close()
    return {
        "checkpoint": os.path.basename(checkpoint) if checkpoint != CONTROL else CONTROL,
        "state": state, "placements": placements,
        "delta_lines": int(info.get(lines_var, 0) or 0) - lines0,
        "end_reason": end, "holes": float(info.get("holes", 0) or 0),
        "height": float(info.get("height", 0) or 0),
        "env_steps": frames, "emulator_frames": emu_frames,
        "seconds": round(elapsed, 2),
        "frames_per_second": round(emu_frames / elapsed, 1),
        "placements_per_second": round(placements / elapsed, 2),
        "placement_cap": cap, "deterministic": bool(deterministic),
    }


def cache_path(key):
    return os.path.join(CACHE, key + ".json")


def write_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)            # atomic: no half-written entry can look complete


def key_for(checkpoint_sha, state, cap, player, players, cfg_hash, commit, rom):
    state_sha = next(s["sha256"] for s in manifest()["states"] if s["name"] == state)
    return ids.run_key(checkpoint_sha, state_sha, cfg_hash, commit, rom, player, players,
                       True, cap), state_sha


# --------------------------------------------------------------- launcher --
def evaluate_pairs(pairs, cap, player, players, cfg_hash, commit, rom, workers, verbose=True):
    """Run (checkpoint, state) pairs as isolated subprocesses, honouring cache."""
    todo, results = [], {}
    for ckpt, state in pairs:
        sha = CONTROL if ckpt["path"] == CONTROL else ckpt["sha256"]
        key, _ = key_for(sha, state, cap, player, players, cfg_hash, commit, rom)
        path = cache_path(key)
        if os.path.exists(path):
            results[(ckpt["name"], state)] = json.load(open(path))
        else:
            todo.append((ckpt, state, key))
    if verbose:
        print("%d pairs: %d cached, %d to run on %d workers"
              % (len(pairs), len(results), len(todo), workers))

    running, done, started = [], 0, time.time()
    while todo or running:
        while todo and len(running) < workers:
            ckpt, state, key = todo.pop(0)
            work = tempfile.mkdtemp(prefix="prog_run_")     # isolated per pair
            cmd = [sys.executable, os.path.abspath(__file__), "--one",
                   "--checkpoint", ckpt["path"], "--state", state,
                   "--cap", str(cap), "--player", str(player), "--key", key]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            running.append((subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, env=env),
                            ckpt, state, key, work))
        for entry in list(running):
            proc, ckpt, state, key, work = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            output = proc.stdout.read() if proc.stdout else ""
            path = cache_path(key)
            if proc.returncode != 0 or not os.path.exists(path):
                print("  FAILED %s on %s (exit %s): %s"
                      % (ckpt["name"], state, proc.returncode,
                         " | ".join(output.strip().splitlines()[-2:])))
            else:
                res = json.load(open(path))
                results[(ckpt["name"], state)] = res
                done += 1
                if verbose:
                    print("  %-6s %-8s placements %4d  delta lines %4d  %-14s %6.0f fps"
                          % (ckpt["name"], state, res["placements"], res["delta_lines"],
                             res["end_reason"], res["frames_per_second"]))
            try:
                os.rmdir(work)
            except OSError:
                pass
        time.sleep(0.05)
    if verbose and done:
        print("ran %d pairs in %.0fs" % (done, time.time() - started))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--one", action="store_true", help="worker mode: run a single pair")
    p.add_argument("--checkpoint", help="worker mode: checkpoint path, or 'control'")
    p.add_argument("--state", help="worker mode: state name")
    p.add_argument("--key", help="worker mode: cache key to write")
    p.add_argument("--checkpoints", default="control",
                   help="'control', 'all', or a comma list of step counts e.g. 100000,200000")
    p.add_argument("--folder", default=os.path.join(ROOT, "checkpoints", "tetris_v54"))
    p.add_argument("--probe", type=int, default=0,
                   help="run this many pairs spread across the suite and stop")
    p.add_argument("--cap", type=int, default=0, help="placement cap (default: the manifest's)")
    p.add_argument("--player", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()

    m = manifest()
    cap = a.cap or int(m["placement_cap"])
    player = a.player or int(m["player"])

    if a.one:
        result = run_one(a.checkpoint, a.state, cap, player)
        if result["frames_per_second"] < MIN_FPS:
            result["warning"] = ("only %.0f emulator frames/s: expected uncapped execution"
                                 % result["frames_per_second"])
        write_atomic(cache_path(a.key), result)
        print(json.dumps(result))
        return

    import custom_integrations
    custom_integrations.register()
    from rl.env import TrainingSpec
    spec = TrainingSpec(m["game"], {"player": player})
    cfg_hash, commit, rom = ids.config_hash(spec), ids.code_commit(), ids.rom_hash(m["game"])
    if cfg_hash != m["config_sha256"]:
        print("WARNING: resolved config differs from the frozen manifest.")
        print("  manifest %s\n  now      %s" % (m["config_sha256"], cfg_hash))
        print("  Results measured now are NOT comparable with earlier ones.")

    states = [s["name"] for s in m["states"]]
    if a.checkpoints == CONTROL:
        chosen = [{"name": CONTROL, "path": CONTROL, "sha256": CONTROL, "steps": -1}]
    else:
        every = checkpoint_list(a.folder)
        if a.checkpoints == "all":
            chosen = every
        else:
            want = {int(x) for x in a.checkpoints.split(",")}
            chosen = [c for c in every if c["steps"] in want]
        print("%d checkpoints from %s" % (len(chosen), a.folder))
        for c in chosen:
            print("  %-7s %s  %s" % (c["name"], c["sha256"][:16], os.path.basename(c["path"])))

    pairs = [(c, s) for c in chosen for s in states]
    if a.probe:
        step = max(1, len(pairs) // a.probe)
        pairs = pairs[::step][:a.probe]
        print("PROBE: %d pairs at cap %d" % (len(pairs), cap))

    results = evaluate_pairs(pairs, cap, player, m["players"], cfg_hash, commit, rom, a.workers)

    capped = sum(1 for r in results.values() if r["end_reason"] == "placement_cap")
    print("\n%d results, %d reached the placement cap (%d%%)"
          % (len(results), capped, round(100 * capped / max(1, len(results)))))
    if capped and a.probe:
        print("A high cap rate censors the strongest checkpoints: reconsider the cap "
              "before the full sweep.")


if __name__ == "__main__":
    main()
