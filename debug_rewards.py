#!/usr/bin/env python3
"""
Verify the TRAINING reward pipeline end to end -- no guessing.

Builds the exact same wrapped env stack train.py trains on (same make_env,
same flags) and runs three scripted policies through it:

    stand     -- always no-op
    run       -- always RIGHT+B
    run+jump  -- RIGHT+B with a periodic full forward jump

and prints each one's per-decision shaped rewards and episode totals. What the
totals MUST show if the shaping is healthy:

    run  >>  stand          (progress actually pays)
    run+jump >= run         (jumping over the first obstacle pays even more)

If `stand` scores about the same as `run`, the progress signal is dead inside
the training stack (whatever diagnostics said about raw RAM) and there's a
plumbing bug to hunt. If `run` clearly wins here but a trained agent still
converges to standing, the reward is fine and the problem is optimization
(e.g. value-function blowup from unnormalized returns), which is fixed in
train.py by reward normalization, not by touching the shaping.

Usage (pass the SAME shaping flags you train with):
    python debug_rewards.py --game SuperMarioBros3-Nes-v0 \\
        --progress-address 0x053C --powerup-address 0x00ED \\
        --progress-scale 1.0 --death-penalty 25 --jump-bonus 0.3
"""
import argparse

from train import ACTION_TABLE, make_env

NOOP = 0
RUN_RIGHT = 3        # ["RIGHT", "B"]
JUMP_RIGHT_LONG = 8  # ["RIGHT", "A"] held 20 frames
JUMP_LEFT_LONG = 10  # ["LEFT", "A"] held 20 frames -- the observed exploit:
                     # at the left screen edge this moves nowhere, so it tests
                     # whether pure jump-spam earns anything


def run_policy(env, pick_action, label, max_decisions=600, show_first=40):
    obs, info = env.reset()
    total = 0.0
    rewards = []
    for i in range(max_decisions):
        obs, r, terminated, truncated, info = env.step(pick_action(i))
        total += r
        rewards.append(r)
        if terminated or truncated:
            break
    print(f"\n--- {label}: {len(rewards)} decisions, TOTAL shaped reward = {total:+.2f} ---")
    print(f"first {min(show_first, len(rewards))} per-decision rewards:")
    line = "  "
    for i, r in enumerate(rewards[:show_first]):
        line += f"{r:+7.2f}"
        if (i + 1) % 8 == 0:
            print(line)
            line = "  "
    if line.strip():
        print(line)
    return total, len(rewards)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--death-penalty", type=float, default=50.0)
    p.add_argument("--jump-bonus", type=float, default=0.0)
    p.add_argument("--stuck-penalty", type=float, default=0.1)
    p.add_argument("--progress-scale", type=float, default=0.1)
    p.add_argument("--score-bonus", type=float, default=0.01)
    p.add_argument("--life-bonus", type=float, default=25.0)
    p.add_argument("--power-bonus", type=float, default=0.0)
    p.add_argument("--powerup-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--progress-address", type=lambda v: int(v, 0), default=None)
    p.add_argument("--progress-address-high", type=lambda v: int(v, 0), default=None)
    p.add_argument("--progress-add-screen-x", action="store_true")
    p.add_argument("--time-penalty", type=float, default=0.0)
    args = p.parse_args()

    env = make_env(
        args.game, args.state, args.death_penalty, args.jump_bonus,
        stuck_penalty=args.stuck_penalty,
        progress_scale=args.progress_scale, score_bonus=args.score_bonus,
        time_penalty=args.time_penalty, life_bonus=args.life_bonus,
        power_bonus=args.power_bonus, powerup_address=args.powerup_address,
        progress_address=args.progress_address,
        progress_address_high=args.progress_address_high,
        progress_add_screen_x=args.progress_add_screen_x,
    )()

    print(f"Action indices used: NOOP={NOOP} {ACTION_TABLE[NOOP]}, "
          f"RUN={RUN_RIGHT} {ACTION_TABLE[RUN_RIGHT]}, "
          f"JUMP={JUMP_RIGHT_LONG} {ACTION_TABLE[JUMP_RIGHT_LONG]}")

    stand_total, stand_n = run_policy(env, lambda i: NOOP, "STAND (always no-op)")
    run_total, run_n = run_policy(env, lambda i: RUN_RIGHT, "RUN (always RIGHT+B)")
    runjump_total, runjump_n = run_policy(
        env, lambda i: JUMP_RIGHT_LONG if i % 6 == 5 else RUN_RIGHT,
        "RUN+JUMP (RIGHT+B, jump every 6th)")
    jumpleft_total, jumpleft_n = run_policy(
        env, lambda i: JUMP_LEFT_LONG,
        "JUMP-LEFT (the observed left-edge jump-spam exploit)")
    env.close()

    print("\n================= SUMMARY =================")
    print(f"  STAND     : {stand_total:+9.2f}  over {stand_n} decisions")
    print(f"  RUN       : {run_total:+9.2f}  over {run_n} decisions")
    print(f"  RUN+JUMP  : {runjump_total:+9.2f}  over {runjump_n} decisions")
    print(f"  JUMP-LEFT : {jumpleft_total:+9.2f}  over {jumpleft_n} decisions")
    print("===========================================")

    if jumpleft_total > stand_total + 5:
        print("\nWARNING: JUMP-LEFT out-earns STAND -- some per-action bonus is")
        print("still farmable at the left edge. Check --jump-bonus is 0 and that")
        print("no other term pays for motionless actions.")

    if run_total > stand_total + 20:
        print("\nVERDICT: progress pays and standing loses -- the reward pipeline is")
        print("healthy. If a trained agent still converges to standing, the problem")
        print("is optimization stability, not shaping (train.py's reward")
        print("normalization addresses that).")
    else:
        print("\nVERDICT: RUN did not clearly beat STAND -- the progress signal is")
        print("NOT paying inside the training stack. Paste this output to Claude;")
        print("that's a plumbing bug (check --progress-address was passed and that")
        print("the RAM read works in this wrapper stack).")


if __name__ == "__main__":
    main()
