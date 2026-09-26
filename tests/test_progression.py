"""Rules of the checkpoint-progression evaluation that must not drift."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.progression import chart, ids, narrate, report, runners, select, speech, video  # noqa: E402

BASE = dict(checkpoint_sha="a", state_sha="b", cfg_hash="c", commit="d", rom="e",
            player=2, players=2, deterministic=True, placement_cap=500)


class CacheKeyTests(unittest.TestCase):
    def test_same_inputs_same_key(self):
        self.assertEqual(ids.run_key(**BASE), ids.run_key(**BASE))

    def test_every_input_invalidates_the_cache(self):
        # A cached result must never survive a change to any of these: that is
        # how checkpoints end up compared under different rules.
        for field, other in (("checkpoint_sha", "z"), ("state_sha", "z"), ("cfg_hash", "z"),
                             ("commit", "z"), ("rom", "z"), ("player", 1), ("players", 1),
                             ("deterministic", False), ("placement_cap", 300)):
            changed = dict(BASE, **{field: other})
            self.assertNotEqual(ids.run_key(**BASE), ids.run_key(**changed), field)

    def test_object_hash_ignores_key_order_only(self):
        self.assertEqual(ids.sha256_obj({"a": 1, "b": 2}), ids.sha256_obj({"b": 2, "a": 1}))
        self.assertNotEqual(ids.sha256_obj({"a": 1}), ids.sha256_obj({"a": 2}))

    def test_file_hash_reads_content(self):
        with tempfile.TemporaryDirectory() as d:
            one, two = os.path.join(d, "1"), os.path.join(d, "2")
            open(one, "wb").write(b"x" * 10)
            open(two, "wb").write(b"x" * 10)
            self.assertEqual(ids.sha256_file(one), ids.sha256_file(two))
            open(two, "wb").write(b"y" * 10)
            self.assertNotEqual(ids.sha256_file(one), ids.sha256_file(two))


class SelectionTests(unittest.TestCase):
    def results(self, placements):
        return {"s%02d" % i: {"placements": p} for i, p in enumerate(placements, start=1)}

    def test_hard_is_the_tenth_percentile_and_median_is_rank_16(self):
        # 32 states, placements 100..131: hardest is s01, rank 4 is s04,
        # rank 16 is s16.
        chosen = select.select(self.results(range(100, 132)))
        self.assertEqual(chosen["hard_state"], "s04")
        self.assertEqual(chosen["median_state"], "s16")

    def test_ranking_is_ascending_by_placements(self):
        chosen = select.select(self.results([130, 101, 120] + list(range(200, 229))))
        order = chosen["ranking_ascending_by_control_placements"]
        self.assertEqual(order[0], "s02")          # 101, the fewest placements
        self.assertEqual(order[1], "s03")          # 120

    def test_ties_break_on_name_so_the_choice_is_reproducible(self):
        tied = {"b": {"placements": 5}, "a": {"placements": 5}, "c": {"placements": 9}}
        self.assertEqual(select.rank_states(tied), ["a", "b", "c"])

    def test_median_never_equals_hard(self):
        chosen = select.select(self.results([7, 9]))
        self.assertNotEqual(chosen["median_state"], chosen["hard_state"])

    def test_empty_control_results_refuse_to_choose(self):
        with self.assertRaises(ValueError):
            select.select({})


class ReportTests(unittest.TestCase):
    def rows(self, cap_500=2, cap_2000=3):
        out = [{"checkpoint": "ckpt_100000_steps.zip", "state": "s%d" % i, "placements": 500,
                "delta_lines": 195, "end_reason": "placement_cap", "placement_cap": 500}
               for i in range(cap_500)]
        out += [{"checkpoint": "ckpt_100000_steps.zip", "state": "s%d" % i, "placements": 700,
                 "delta_lines": 260, "end_reason": "game_over", "placement_cap": 2000}
                for i in range(cap_2000)]
        return out

    def test_results_from_different_caps_are_never_pooled(self):
        # Mixing budgets compares checkpoints under different rules, which is
        # what the whole design guards against.
        with tempfile.TemporaryDirectory() as d:
            for i, row in enumerate(self.rows()):
                json.dump(row, open(os.path.join(d, "%02d.json" % i), "w"))
            original, report.CACHE = report.CACHE, d
            try:
                kept, excluded = report.load_results(2000)
            finally:
                report.CACHE = original
            self.assertEqual(len(kept), 3)
            self.assertEqual(excluded, 2)
            self.assertTrue(all(r["placement_cap"] == 2000 for r in kept))

    def test_capped_trajectories_are_counted_not_hidden(self):
        summary = report.summarise([
            {"checkpoint": "c", "delta_lines": 10, "end_reason": "game_over", "placement_cap": 2000},
            {"checkpoint": "c", "delta_lines": 99, "end_reason": "placement_cap", "placement_cap": 2000},
        ])
        self.assertEqual(summary["c"]["capped_trajectories"], 1)
        self.assertEqual(summary["c"]["states"], 2)

    def test_distribution_has_no_threshold_field(self):
        summary = report.summarise([
            {"checkpoint": "c", "delta_lines": v, "end_reason": "game_over", "placement_cap": 2000}
            for v in (10, 20, 30, 40)])
        self.assertEqual(sorted(summary["c"]), sorted(
            ["states", "median", "q1", "q3", "min", "max", "capped_trajectories",
             "steps", "placement_cap", "iqr"]))
        self.assertEqual(summary["c"]["median"], 25.0)
        self.assertEqual(summary["c"]["iqr"], 15.0)

    def test_checkpoint_labels_sort_control_first_then_by_steps(self):
        self.assertEqual(report.order_labels(["200k", "control", "5k", "100k"]),
                         ["control", "5k", "100k", "200k"])


class ManifestTests(unittest.TestCase):
    PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "progression_out", "suite_manifest.json")

    def manifest(self):
        if not os.path.exists(self.PATH):
            self.skipTest("suite not generated in this checkout")
        return json.load(open(self.PATH))

    def test_manifest_hash_matches_its_recorded_digest(self):
        digest_file = self.PATH + ".sha256"
        if not os.path.exists(digest_file):
            self.skipTest("suite not generated in this checkout")
        body = open(self.PATH).read()
        self.assertEqual(ids.sha256_text(body), open(digest_file).read().split()[0],
                         "the frozen suite manifest has been edited since it was hashed")

    def test_states_are_behaviourally_distinct_not_just_different_files(self):
        m = self.manifest()
        prefixes = [tuple(s["piece_prefix"]) for s in m["states"]]
        self.assertEqual(len(set(prefixes)), len(prefixes),
                         "two states deal the same pieces: the suite repeats a trial")
        hashes = [s["sha256"] for s in m["states"]]
        self.assertEqual(len(set(hashes)), len(hashes))

    def test_manifest_records_everything_a_result_depends_on(self):
        m = self.manifest()
        for key in ("rom_sha256", "config_sha256", "code_commit", "placement_cap",
                    "player", "players", "selection_rule", "generation"):
            self.assertIn(key, m)
        self.assertNotEqual(m["rom_sha256"], "missing")


if __name__ == "__main__":
    unittest.main()


class ChartStatsTests(unittest.TestCase):
    def rows(self, values, ends=None):
        ends = ends or ["game_over"] * len(values)
        return [{"column": "c", "value": v, "decisions": 100, "end_reason": e}
                for v, e in zip(values, ends)]

    def test_sample_standard_deviation_not_population(self):
        # n-1. The population sd flatters a small sample, and every column
        # here is a handful of start states.
        s = chart.summarise(self.rows([2, 4, 4, 4, 5, 5, 7, 9]))
        self.assertAlmostEqual(s["mean"], 5.0)
        self.assertAlmostEqual(s["sd"], 2.13808, places=4)

    def test_single_game_has_undefined_spread_not_zero(self):
        self.assertIsNone(chart.summarise(self.rows([42]))["sd"])

    def test_capped_runs_are_counted_apart_from_finished_ones(self):
        s = chart.summarise(self.rows([10, 20, 30],
                                      ["game_over", "placement_cap", "game_over"]))
        self.assertEqual((s["finished"], s["capped"]), (2, 1))
        self.assertEqual(s["n"], 3)



class NarrationTimingTests(unittest.TestCase):
    """The rule the schedule exists to keep: nothing about the RESULTS is
    spoken until the results are on screen."""

    def clips(self, lengths):
        return [("block_%d.wav" % i, s) for i, s in enumerate(lengths)]

    def test_body_runs_consecutively_from_the_start(self):
        placed = narrate.schedule(self.clips([10, 10, 10]), [], grid_seconds=60.0)
        self.assertAlmostEqual(placed[0][0], 0.6)
        # One breath between paragraphs, not a gap sized to fill the video.
        self.assertAlmostEqual(placed[1][0] - placed[0][1], narrate.BLOCK_GAP)
        self.assertFalse(any(on_card for _s, _e, _p, on_card in placed))

    def test_no_card_line_is_spoken_before_the_card_is_up(self):
        placed = narrate.schedule(self.clips([5, 5]), self.clips([8, 4]),
                                  grid_seconds=60.0)
        card = [start for start, _e, _p, on_card in placed if on_card]
        self.assertTrue(card)
        self.assertTrue(all(start >= 60.0 for start in card),
                        "a results line would be spoken over a game still playing")

    def test_a_long_body_still_never_pushes_card_lines_early(self):
        # The body overrunning the gameplay must not drag the card part back
        # before the card: the card start is a floor, not an offset.
        placed = narrate.schedule(self.clips([80]), self.clips([5]), grid_seconds=60.0)
        self.assertGreaterEqual([s for s, _e, _p, c in placed if c][0], 80.0)

    def test_the_card_is_held_for_exactly_what_is_said_over_it(self):
        held = narrate.card_seconds(self.clips([8, 6]))
        self.assertGreater(held, 14.0)                 # both blocks fit
        self.assertLess(held, 18.0)                    # and not much more
        self.assertEqual(narrate.card_seconds([]), 0.0)

    def test_facts_name_the_unit_of_every_number(self):
        # A mean with no unit was read aloud as "the average placement count"
        # when it was lines cleared.
        card = {"order": ["100k"], "metric_label": "lines", "work_label": "pieces",
                "stats": {"100k": {"n": 2, "mean": 496.0, "sd": 422.0, "min": 198.0,
                                   "max": 795.0, "median": 496.0, "decisions": 1265.0,
                                   "per_100": 39.0, "finished": 1, "capped": 1}}}
        text = narrate.facts("Tetris", card, ["a", "b"], 2000)
        self.assertIn("mean 496 lines per game", text)
        self.assertIn("standard deviation 422 lines", text)
        self.assertIn("averaging 1265 pieces per game", text)
        self.assertIn("The headline number for each game is lines", text)

    def test_facts_survive_a_card_that_does_not_name_its_units(self):
        card = {"order": ["c"], "stats": {"c": {"n": 1, "mean": 5.0, "sd": None,
                                                "min": 5.0, "max": 5.0, "median": 5.0,
                                                "decisions": 10.0, "per_100": 50.0,
                                                "finished": 1, "capped": 0}}}
        text = narrate.facts("G", card, ["a"], 100)
        self.assertIn("no spread to report from a single game", text)


class SpeechBackendTests(unittest.TestCase):
    def test_unknown_backend_is_refused_by_name(self):
        with self.assertRaises(ValueError):
            speech.speak(["hello"], "/tmp", backend="not-a-real-tts")

    def test_a_missing_backend_fails_before_anything_is_rendered(self):
        # The check exists so a voice that is not installed costs nothing --
        # it must raise rather than return empty.
        if speech.available("piper"):
            self.skipTest("piper is installed here")
        with self.assertRaises(RuntimeError):
            speech.speak(["hello"], "/tmp", backend="piper")


if __name__ == "__main__":
    unittest.main()


class CatchUpTests(unittest.TestCase):
    """Playing to game over means one survivor can hold the film open while
    everything else sits frozen. The rate plan is what stops that."""

    LENGTHS = [100.0, 120.0, 300.0, 900.0]

    def test_disabled_is_one_constant_rate(self):
        plan, out = video.rate_plan(self.LENGTHS, 8.0, catch_up=1.0)
        self.assertTrue(all(rate == 8.0 for _end, rate in plan))
        self.assertAlmostEqual(out, 900.0 / 8.0)

    def test_rate_rises_only_as_games_end_and_never_falls(self):
        plan, _out = video.rate_plan(self.LENGTHS, 8.0, catch_up=6.0)
        rates = [rate for _end, rate in plan]
        self.assertEqual(rates[0], 8.0, "it must start at the speed asked for")
        self.assertEqual(rates, sorted(rates), "the film must never slow down")
        # Boundaries are exactly the moments a game ends, so the speed never
        # changes in the middle of the action.
        self.assertEqual([end for end, _rate in plan], sorted(set(self.LENGTHS)))

    def test_boost_is_proportional_to_how_many_are_still_playing(self):
        plan, _out = video.rate_plan(self.LENGTHS, 8.0, catch_up=99.0)
        # 4 alive, then 3, 2, 1 -> 1x, 4/3, 2x, 4x of the base rate.
        self.assertEqual([round(r / 8.0, 3) for _e, r in plan], [1.0, 1.333, 2.0, 4.0])

    def test_catch_up_is_a_ceiling(self):
        plan, _out = video.rate_plan(self.LENGTHS, 8.0, catch_up=2.0)
        self.assertEqual(max(r for _e, r in plan), 16.0)

    def test_identical_games_get_no_boost(self):
        plan, out = video.rate_plan([200.0, 200.0, 200.0], 4.0, catch_up=6.0)
        self.assertEqual(len(plan), 1)
        self.assertAlmostEqual(out, 50.0)

    def test_total_is_the_sum_of_the_segments(self):
        plan, out = video.rate_plan(self.LENGTHS, 8.0, catch_up=6.0)
        starts = [0.0] + [end for end, _r in plan[:-1]]
        self.assertAlmostEqual(
            out, sum((end - start) / rate
                     for (end, rate), start in zip(plan, starts)))
        self.assertLess(out, 900.0 / 8.0, "catch-up must shorten the film")

    def test_the_expression_escapes_its_commas(self):
        # An unescaped comma is a FILTER SEPARATOR in a filtergraph: ffmpeg
        # went looking for a filter called 'min(T' and the render died.
        expr = video.remap(video.rate_plan(self.LENGTHS, 8.0, 6.0)[0])
        self.assertIn("\,", expr)
        self.assertNotIn("min(T,", expr)
        self.assertTrue(expr.startswith("setpts=("))


class HoldTests(unittest.TestCase):
    def test_hold_adds_exactly_its_seconds_at_real_speed(self):
        plan_no, out_no = video.rate_plan([100.0, 200.0], 8.0, 4.0, hold=0.0)
        plan_yes, out_yes = video.rate_plan([100.0, 200.0], 8.0, 4.0, hold=3.0)
        self.assertAlmostEqual(out_yes - out_no, 3.0)
        # Real time, so three seconds on screen is three seconds of game.
        self.assertEqual(plan_yes[-1][1], 1.0)
        self.assertEqual(len(plan_yes), len(plan_no) + 1)

    def test_no_hold_adds_no_segment(self):
        plan, _out = video.rate_plan([100.0, 200.0], 8.0, 4.0, hold=0.0)
        self.assertTrue(all(rate >= 8.0 for _end, rate in plan))


class CounterWrapTests(unittest.TestCase):
    """A HUD counter has a fixed number of digits and rolls over. A 100k
    checkpoint that cleared 1057 lines was recorded as 57, which ranked it
    below checkpoints it had beaten several times over."""

    class FakeVars:
        def __init__(self, spec):
            self.spec = spec

    def test_modulus_comes_from_the_declared_digit_width(self):
        gv = self.FakeVars({"lines": {"source": "tiles", "length": 3}})
        self.assertEqual(runners.counter_modulus(gv, "lines_p1", "lines"), 1000)

    def test_unknown_encoding_means_no_correction(self):
        gv = self.FakeVars({"lines": {"source": "ram", "address": "0x1234"}})
        self.assertEqual(runners.counter_modulus(gv, "lines"), 0)
        self.assertEqual(runners.counter_modulus(self.FakeVars({}), "nope"), 0)

    def test_a_four_digit_counter_wraps_at_ten_thousand(self):
        gv = self.FakeVars({"score": {"source": "tiles", "length": 4}})
        self.assertEqual(runners.counter_modulus(gv, "score"), 10000)


class CardFitTests(unittest.TestCase):
    """A results card with nothing said over it is the one outcome that is
    never acceptable: the winner goes unannounced."""

    def test_the_winner_line_and_the_challenge_both_survive_a_tight_cap(self):
        clips = [("win.wav", 22.5), ("mid.wav", 12.0), ("end.wav", 8.6)]
        kept, index, hold = narrate.fit_card(clips, 22.0)
        self.assertEqual(index[0], 0, "the winner line must be kept")
        self.assertEqual(index[-1], len(clips) - 1, "the challenge must be kept")
        self.assertNotIn(1, index, "the middle is what the cap trims")
        self.assertGreater(hold, 22.0, "held long enough to say both, cap or not")

    def test_everything_fits_when_there_is_room(self):
        clips = [("a.wav", 9.0), ("b.wav", 7.0), ("c.wav", 6.0)]
        kept, index, hold = narrate.fit_card(clips, 30.0)
        self.assertEqual(index, [0, 1, 2])
        self.assertAlmostEqual(hold, narrate.card_seconds(clips))

    def test_the_hold_always_matches_what_is_kept(self):
        clips = [("a.wav", 10.0), ("b.wav", 10.0), ("c.wav", 10.0), ("d.wav", 5.0)]
        kept, _index, hold = narrate.fit_card(clips, 26.0)
        self.assertAlmostEqual(hold, narrate.card_seconds(kept))

    def test_no_card_means_no_hold(self):
        self.assertEqual(narrate.fit_card([], 22.0), ([], [], 0.0))


class LayoutTests(unittest.TestCase):
    """The grid shape is DERIVED from the panel's own aspect and the canvas,
    because intuition gets it wrong."""

    TETRIS = (152, 224)          # a well, taller than it is wide
    MARIO = (240, 224)           # a whole screen, wider than tall

    def test_two_tall_panels_stack_in_a_phone_frame(self):
        # Side by side wastes the height: 540x794 each against 652x960 stacked.
        self.assertEqual(video.best_grid(2, *self.TETRIS, 1080, 1826), 1)

    def test_the_same_two_sit_side_by_side_in_a_wide_frame(self):
        self.assertEqual(video.best_grid(2, *self.TETRIS, 1920, 986), 2)

    def test_fifteen_panels_land_on_the_shape_that_was_chosen_by_hand(self):
        self.assertEqual(video.best_grid(15, *self.TETRIS, 1920, 986), 5)

    def test_a_wide_game_stacks_in_portrait_too(self):
        self.assertEqual(video.best_grid(2, *self.MARIO, 1080, 1826), 1)

    def test_one_panel_is_one_column(self):
        self.assertEqual(video.best_grid(1, *self.TETRIS, 1080, 1826), 1)
        self.assertEqual(video.best_grid(1, *self.MARIO, 1920, 986), 1)

    def test_the_choice_actually_maximises_the_panel(self):
        # Whatever it picks must beat every other arrangement on rendered area.
        w, h, cw, ch = 152, 224, 1080, 1826
        chosen = video.best_grid(6, w, h, cw, ch)

        def area(cols):
            rows = -(-6 // cols)
            cell_w = (cw - (cols - 1) * 48) / cols
            cell_h = (ch - (rows - 1) * 48) / rows
            return min(cell_w / w, cell_h / h) ** 2 * w * h

        self.assertEqual(chosen, max(range(1, 7), key=area))
