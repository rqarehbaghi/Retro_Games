import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from narrate_ensemble import validate, narration_schema


class EnsembleValidationTest(unittest.TestCase):
    def setUp(self):
        self.evidence = {'segments': [{'id': 's1', 'max_words': 4,
                                      'observations': [{'id': 'e1', 'text': 'Health falls'}]}]}

    def test_valid_passage(self):
        script = {'segments': [{'slot': 's1', 'text': 'That leaves less room.', 'evidence_ids': ['e1']}]}
        self.assertEqual(validate(script, self.evidence), script)

    def test_reject_wrong_evidence_and_excess_words(self):
        for text, refs in [('Okay.', ['invented']), ('One two three four five', ['e1'])]:
            with self.assertRaises(ValueError):
                validate({'segments': [{'slot': 's1', 'text': text, 'evidence_ids': refs}]}, self.evidence)

    def test_schema_forbids_extra_outro(self):
        schema = narration_schema(self.evidence)['properties']['segments']
        self.assertEqual(schema['minItems'], 1)
        self.assertEqual(schema['maxItems'], 1)
        self.assertEqual(schema['items']['properties']['slot']['enum'], ['s1'])
        with self.assertRaisesRegex(ValueError, 'No extra outro'):
            validate({'segments': []}, self.evidence)
