"""Rules of the checkpoint-progression evaluation that must not drift."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.progression import chart, ids, narrate, report, select   # noqa: E402

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
    def lines(self, lengths, chart_flags=None, closing=None):
        flags = chart_flags or [False] * len(lengths)
        out = [{"text": "l%d" % i, "seconds": s, "chart": c, "closing": False}
               for i, (s, c) in enumerate(zip(lengths, flags))]
        if closing:
            out.append({"text": "end", "seconds": closing, "chart": True, "closing": True})
        return out

    def test_lines_are_spread_across_the_whole_video_not_packed_at_the_front(self):
        # The point of the schedule: 3 x 5s of speech in 60s of video must not
        # finish at 0:16 and leave 44 seconds of silence.
        placed, dropped = narrate.spread(self.lines([5, 5, 5]), 0.6, 60.0)
        self.assertEqual(dropped, [])
        self.assertGreater(placed[-1][0], 30.0)
        self.assertLessEqual(placed[-1][1], 60.0)

    def test_speech_longer_than_the_window_is_dropped_not_pushed_past_the_end(self):
        placed, dropped = narrate.spread(self.lines([5, 5, 5]), 0.6, 8.0)
        self.assertEqual(len(placed), 1)
        self.assertEqual(len(dropped), 2)

    def test_chart_lines_wait_for_the_card_and_the_closing_lands_last(self):
        lines = self.lines([4, 4, 3], [False, False, True], closing=3)
        placed, _ = narrate.plan(lines, seconds=60.0, chart_at=50.0)
        by_text = {item["text"]: (start, end) for start, end, _l, item in placed}
        self.assertLess(by_text["l1"][1], 50.0)          # body ends before the card
        self.assertGreaterEqual(by_text["l2"][0], 50.0)  # the card line waits
        self.assertAlmostEqual(by_text["end"][1], 59.7, places=1)

    def test_without_a_card_the_body_uses_the_whole_video(self):
        lines = self.lines([4, 4], closing=3)
        placed, _ = narrate.plan(lines, seconds=60.0, chart_at=60.0)
        body_end = max(end for _s, end, _l, item in placed if not item["closing"])
        self.assertGreater(body_end, 40.0)
