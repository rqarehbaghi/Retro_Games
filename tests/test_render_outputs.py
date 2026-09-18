"""What a studio run renders, which encoder it uses, and how the blur is built.

No ffmpeg run and no GPU needed: the NVENC probe result is injected.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import overlays   # noqa: E402
import studio     # noqa: E402


class OutputTests(unittest.TestCase):
    def test_clean_pair_by_default_overlay_pair_only_on_request(self):
        plain = studio.studio_outputs("run", False)
        self.assertEqual([o["file"] for o in plain],
                         ["run_16x9_clean.mp4", "run_9x16_clean.mp4"])
        self.assertTrue(all(o["overlays"] is False for o in plain))
        overlaid = studio.studio_outputs("run", True)
        self.assertEqual([o["file"] for o in overlaid],
                         ["run_16x9_clean.mp4", "run_9x16_clean.mp4",
                          "run_16x9.mp4", "run_9x16.mp4"])
        self.assertEqual([(o["width"], o["height"]) for o in overlaid],
                         [(1920, 1080), (1080, 1920), (1920, 1080), (1080, 1920)])

    def test_brief_prefers_overlay_videos_and_falls_back_to_clean(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("run_16x9_clean.mp4", "run_9x16_clean.mp4"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual([os.path.basename(p) for _r, p in studio.staged_videos(d)],
                             ["run_16x9_clean.mp4", "run_9x16_clean.mp4"])
            for name in ("run_16x9.mp4", "run_9x16.mp4", "run_16x9_narrated.mp4"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual([os.path.basename(p) for _r, p in studio.staged_videos(d)],
                             ["run_16x9.mp4", "run_9x16.mp4"])


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self._saved = overlays._NVENC

    def tearDown(self):
        overlays._NVENC = self._saved

    def test_default_is_x264_even_when_nvenc_works(self):
        # NVENC measured slower and failed 3 of 18 encodes under WSL2.
        overlays._NVENC = (True, "ok")
        args, _ = overlays.video_codec_args(overlays.merge_style({}))
        self.assertEqual(args, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])

    def test_auto_uses_nvenc_when_the_test_encode_worked(self):
        overlays._NVENC = (True, "ok")
        args, name = overlays.video_codec_args(overlays.merge_style({"encoder": "auto"}))
        self.assertEqual(args[:2], ["-c:v", "h264_nvenc"])
        self.assertIn("NVENC", name)

    def test_auto_falls_back_to_x264_veryfast(self):
        overlays._NVENC = (False, "cuInit(0) failed")
        args, name = overlays.video_codec_args(overlays.merge_style({"encoder": "auto"}))
        self.assertEqual(args, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])

    def test_pinned_nvenc_fails_loudly_instead_of_silently_using_cpu(self):
        overlays._NVENC = (False, "cuInit(0) failed")
        with self.assertRaises(SystemExit):
            overlays.video_codec_args(overlays.merge_style({"encoder": "nvenc"}))

    def test_pinned_x264_never_probes(self):
        overlays._NVENC = (True, "ok")
        args, _ = overlays.video_codec_args(overlays.merge_style({"encoder": "x264"}))
        self.assertEqual(args[1], "libx264")

    def test_old_overlays_json_still_renders(self):
        # A folder staged before these settings existed has only crf in style.
        style = overlays.merge_style({"crf": 20, "blur": 20})
        overlays._NVENC = (False, "no gpu")
        args, _ = overlays.video_codec_args(style)
        self.assertIn("20", args)


class TitleTests(unittest.TestCase):
    def title_stage(self, w, h, style=None):
        spec = {"title": "Tetris Time", "style": style or {}}
        return [p for p in overlays.build_filter(spec, w, h).split(";")
                if "Tetris Time" in p][0]

    def test_title_leaves_after_five_seconds_on_wide_only(self):
        self.assertIn("enable='lt(t,5.00)'", self.title_stage(1920, 1080))
        self.assertNotIn("enable=", self.title_stage(1080, 1920))

    def test_wide_seconds_null_keeps_it_up(self):
        stage = self.title_stage(1920, 1080, {"title": {"wide_seconds": None}})
        self.assertNotIn("enable=", stage)

    def test_all_text_is_on_an_opaque_plate_without_outline(self):
        stage = self.title_stage(1920, 1080)
        self.assertIn("box=1", stage)
        self.assertIn("boxcolor=white:", stage)
        self.assertIn("borderw=0", stage)
        self.assertNotIn("shadowcolor", stage)


class BlurTests(unittest.TestCase):
    def test_fill_is_blurred_small_with_matching_sigma(self):
        spec = {"style": {"blur": 20}}
        wide = overlays.build_filter(spec, 1920, 1080, overlays=False)
        self.assertIn("crop=240:135,gblur=sigma=2.5,scale=1920:1080", wide)
        tall = overlays.build_filter(spec, 1080, 1920, overlays=False)
        self.assertIn("crop=135:240,gblur=sigma=2.5,scale=1080:1920", tall)
        self.assertTrue(wide.endswith("[v0]null[vout]"))


class OversampleTests(unittest.TestCase):
    def test_sharp_oversampling_is_used_by_default(self):
        spec = {}
        wide = overlays.build_filter(spec, 1920, 1080, overlays=False)
        self.assertIn("scale=iw*8:ih*8:flags=neighbor,scale=1920:1080:force_original_aspect_ratio=decrease:flags=area", wide)

    def test_oversampling_disabled_when_one_or_less(self):
        spec = {"style": {"oversample": 1}}
        wide = overlays.build_filter(spec, 1920, 1080, overlays=False)
        self.assertIn("scale=1920:1080:force_original_aspect_ratio=decrease:flags=neighbor", wide)


if __name__ == "__main__":
    unittest.main()
