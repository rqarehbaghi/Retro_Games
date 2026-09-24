"""Which two states the video shows. Decided from the CONTROL only.

Written before any checkpoint ran. The rule is fixed here so that it cannot be
adjusted once results exist -- picking states after seeing checkpoint results
would make the grid a flattering selection with a disclaimer, which is the
failure mode this whole design is trying to avoid.
"""

import math

MEDIAN_RANK = 16          # of 32, upper median, 1-based on the ascending ranking
HARD_QUANTILE = 0.10      # 10th percentile: rank 4 of 32


def rank_states(control_results):
    """States hardest-first by the control's placements, ties by name.

    `control_results` maps state name -> dict with "placements". Ascending
    placements means the control survived least, i.e. hardest first."""
    return sorted(control_results,
                  key=lambda name: (control_results[name]["placements"], name))


def hard_rank(count, quantile=HARD_QUANTILE):
    """1-based rank for `quantile`, at least 1 and at most `count`.

    CEILING, not rounding: the published plan fixed the 10th percentile of 32
    states as rank 4, and 0.10 * 32 = 3.2 rounds to 3. The rule was declared
    before any result existed, so the code follows the rule."""
    return max(1, min(count, math.ceil(quantile * count)))


def select(control_results, median_rank=MEDIAN_RANK, quantile=HARD_QUANTILE):
    """(median_state, hard_state) plus the ranking, for the manifest.

    Both come from the same ascending ranking: rank 1 is where the control did
    worst. If the two land on the same state -- possible only on a tiny suite
    -- the median moves to the next state so the video never shows one state
    twice."""
    order = rank_states(control_results)
    if not order:
        raise ValueError("no control results: the control must run before states are chosen")
    hard = order[hard_rank(len(order), quantile) - 1]
    median = order[min(median_rank, len(order)) - 1]
    if median == hard:
        remaining = [s for s in order if s != hard]
        if not remaining:
            raise ValueError("need at least two states to choose from")
        median = remaining[min(median_rank, len(remaining)) - 1]
    return {
        "median_state": median,
        "hard_state": hard,
        "ranking_ascending_by_control_placements": order,
        "rule": ("rank %d of %d for the median state and rank %d (%.0fth percentile) for the "
                 "hard state, both on the zero-value control's placements ascending, "
                 "ties broken on state name" %
                 (min(median_rank, len(order)), len(order),
                  hard_rank(len(order), quantile), quantile * 100)),
    }
