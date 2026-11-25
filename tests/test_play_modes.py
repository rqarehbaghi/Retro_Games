"""Exercise pure routing helpers without initializing pygame or an emulator."""
import ast
from pathlib import Path
import unittest

source = ast.parse((Path(__file__).resolve().parents[1] / 'play_engine.py').read_text(encoding='utf-8'))
helpers = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef)
    and node.name in ('ai_requested', 'configured_repress_index')], type_ignores=[])
namespace = {}
exec(compile(helpers, 'play_engine.py', 'exec'), namespace)


class PlayModeTests(unittest.TestCase):
    def test_human_and_model_modes(self):
        requested = namespace['ai_requested']
        self.assertFalse(requested(1, None, False))
        self.assertFalse(requested(2, None, True))
        self.assertTrue(requested(1, 'model.zip', False))
        self.assertTrue(requested(2, None, False))

    def test_placeholder_is_not_configured_button(self):
        index = namespace['configured_repress_index']
        buttons = ['B', None, 'DOWN']
        self.assertIsNone(index(buttons, {}))
        self.assertIsNone(index(buttons, {'button': None}))
        self.assertIsNone(index(buttons, {'button': 'MISSING'}))
        self.assertEqual(index(buttons, {'button': 'DOWN'}), 2)
