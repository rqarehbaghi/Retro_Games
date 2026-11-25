import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import tts as voice


class PaidVoiceTests(unittest.TestCase):
    def test_explicit_authorization_and_key_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                voice.elevenlabs_check_config('voice1', True, 100)
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-only'}):
            with self.assertRaises(ValueError):
                voice.elevenlabs_check_config('voice1', False, 100)
            voice.elevenlabs_check_config('voice1', True, 100)

    def test_paid_preflight_rejects_free_before_generation(self):
        with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-only'}), \
                patch.object(voice, 'elevenlabs_api', return_value=b'{"tier":"free"}') as api:
            with self.assertRaises(ValueError):
                voice.elevenlabs_preflight('voice1', True, 100)
            self.assertEqual(api.call_count, 1)

    def test_character_limit_precedes_api(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-only'}), \
                patch.object(voice, 'elevenlabs_api') as api:
            with self.assertRaises(ValueError):
                voice.elevenlabs_synthesize(['too many characters'], directory, 'voice1', allow_paid=True, max_chars=2)
            api.assert_not_called()

    def test_ensemble_import_and_bad_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'evidence': {'segments': [
                {'id': 's1', 'start': 1, 'end': 3}]}}))
            path = root / 'master.json'
            path.write_text(json.dumps({'segments': [{'slot': 's1', 'text': 'Hello.'}]}))
            self.assertEqual(voice.elevenlabs_load_timed_script(path)[0]['start'], 1)
            path.write_text(json.dumps({'segments': [{'start': 3, 'end': 1, 'text': 'Hello.'}]}))
            with self.assertRaises(ValueError):
                voice.elevenlabs_load_timed_script(path)

    def test_overlong_audio_is_not_clipped_or_shifted(self):
        with patch.object(voice, 'elevenlabs_duration', return_value=3), patch.object(voice.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                voice.elevenlabs_timed_track([{'start': 1, 'end': 2, 'text': 'Hello'}], ['clip.wav'], 5, 'out.wav')
            run.assert_not_called()

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_cached_speech_and_real_mux_with_and_without_game_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'mock.mp3'
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=0.4', str(audio)], check=True)
            with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-only'}), \
                    patch.object(voice, 'elevenlabs_preflight'), \
                    patch.object(voice, 'elevenlabs_api', return_value=audio.read_bytes()) as api:
                clips = voice.elevenlabs_synthesize(['Hello there.'], root / 'cache', 'voice1', allow_paid=True)
                voice.elevenlabs_synthesize(['Hello there.'], root / 'cache', 'voice1', allow_paid=True)
                self.assertEqual(api.call_count, 1)
            track = root / 'track.wav'
            placed = voice.elevenlabs_timed_track([{'start': 0.2, 'end': 1.2, 'text': 'Hello there.'}], clips, 2, track)
            for width, height, with_audio in [(640, 360, True), (360, 640, False)]:
                video = root / ('source_%s.mp4' % width)
                command = ['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                           'color=c=blue:s=%dx%d:r=15:d=2' % (width, height)]
                if with_audio:
                    command += ['-f', 'lavfi', '-i', 'sine=frequency=220:duration=2']
                subprocess.run(command + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
                out = root / ('out_%s.mp4' % width)
                voice.elevenlabs_mux(video, track, out, placed)
                info = voice.elevenlabs_probe(out)
                self.assertAlmostEqual(float(info['format']['duration']), 2, delta=0.15)
                self.assertTrue(any(s['codec_type'] == 'audio' for s in info['streams']))
