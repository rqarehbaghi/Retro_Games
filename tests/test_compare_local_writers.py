import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_local_writers as compare


class CompareWritersTest(unittest.TestCase):
    def test_reasoning_is_not_spoken(self):
        self.assertEqual(compare.clean_script('Draft notes</think> My game.'), ('My game.', True))
        with self.assertRaises(ValueError):
            compare.clean_script('<think>unfinished notes')

    def test_exports_local_and_skips_remote(self):
        def request(host, endpoint, payload=None, timeout=30):
            if endpoint == '/api/tags':
                return {'models': [{'name': n, 'digest': n} for n in ['local', 'remote']]}
            if endpoint == '/api/show':
                return ({'remote_host': 'example.com'} if payload['model'] == 'remote'
                        else {'capabilities': ['completion', 'thinking']})
            self.assertEqual(payload['model'], 'local')
            self.assertFalse(payload['think'])
            self.assertEqual(payload['keep_alive'], 0)
            return {'response': 'Host 1: Hello. Host 2: Welcome.', 'done_reason': 'stop'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = root / 'facts.txt'
            facts.write_text('Recorded facts.', encoding='utf-8')
            out = root / 'output'
            with patch('sys.argv', ['compare', '--facts', str(facts), '--out-dir', str(out)]), \
                    patch.object(compare, 'request', side_effect=request):
                compare.main()
            records = json.loads((out / 'results.json').read_text())
            self.assertTrue((out / records[0]['file']).exists())
            self.assertIn('skipped', records[1])
            self.assertEqual(len(list(out.glob('*.json'))), 2)

    def test_empty_response_is_error_not_reasoning(self):
        def request(host, endpoint, payload=None, timeout=30):
            if endpoint == '/api/tags':
                return {'models': [{'name': 'local'}]}
            if endpoint == '/api/show':
                return {'capabilities': ['completion']}
            return {'response': '', 'thinking': 'private reasoning'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = root / 'facts.txt'
            facts.write_text('Facts', encoding='utf-8')
            out = root / 'output'
            with patch('sys.argv', ['compare', '--facts', str(facts), '--out-dir', str(out)]), \
                    patch.object(compare, 'request', side_effect=request), \
                    self.assertRaises(SystemExit):
                compare.main()
            self.assertIn('error', json.loads((out / 'results.json').read_text())[0])


if __name__ == '__main__':
    unittest.main()
