import unittest
from tools.analyze_gameplay import validate


class VisualEvidenceTest(unittest.TestCase):
    def test_frame_times_come_from_capture_not_model(self):
        value = validate({'observations': [{'frame_indices': [0, 2],
            'description': 'The displayed number changes.', 'confidence': 'medium'}],
            'uncertainties': ['Digits are small.']}, [1, 2, 3])
        self.assertEqual(value['observations'][0]['sample_times_seconds'], [1, 3])

    def test_invalid_frame_reference_rejected(self):
        with self.assertRaises(ValueError):
            validate({'observations': [{'frame_indices': [5], 'description': 'x',
                'confidence': 'high'}], 'uncertainties': []}, [1, 2, 3])
