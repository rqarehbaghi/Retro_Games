"""Identity of a run: what has to be the same for two results to be comparable.

A cache that keys only on "which checkpoint, which state" silently survives a
change to the reward config, the code, or the ROM, and then compares
checkpoints that were never measured under the same rules. Every key here is a
hash of a thing that can change the outcome.
"""
import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj):
    """Stable hash of a JSON-able object: sorted keys, no incidental spacing."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def code_commit():
    """The commit the evaluator is running, plus whether the tree is dirty.

    A dirty tree is recorded rather than refused: the sweep is long, and
    silently comparing results from different working trees is the real risk."""
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head + ("+dirty" if dirty else "")
    except Exception:                                             # noqa: BLE001
        return "unknown"


def rom_hash(game):
    import custom_integrations
    folder = os.path.join(custom_integrations.INTEGRATIONS_DIR, game)
    for name in sorted(os.listdir(folder)):
        if name.startswith("rom."):
            return sha256_file(os.path.join(folder, name))
    return "missing"


def config_hash(spec):
    """The resolved training configuration this evaluation depends on."""
    return sha256_obj({
        "afterstate": getattr(spec, "afterstate_config", {}),
        "grid": getattr(spec, "grid", {}),
        "macro": getattr(spec, "macro_config", {}),
        "skip_while": getattr(spec, "skip_while", None),
        "episode_end": getattr(spec, "episode_end", None),
        "actions": getattr(spec, "actions", None),
        "player": getattr(spec, "player", None),
        "players": getattr(spec, "players", None),
        "features": getattr(spec, "features_name", None),
        "observation": getattr(spec, "observation", {}),
    })


def run_key(checkpoint_sha, state_sha, cfg_hash, commit, rom, player, players,
            deterministic, placement_cap):
    """One cache key. Every argument can change the result."""
    return sha256_obj({
        "checkpoint": checkpoint_sha, "state": state_sha, "config": cfg_hash,
        "commit": commit, "rom": rom, "player": player, "players": players,
        "deterministic": bool(deterministic), "cap": int(placement_cap),
        "schema": 1,
    })
