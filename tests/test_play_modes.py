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


class StudioControllerTests(unittest.TestCase):
    def test_controller_matrix(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'studio.py').read_text())
        helper = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name == 'resolve_controllers'], type_ignores=[])
        scope = {}
        exec(compile(helper, 'studio.py', 'exec'), scope)
        resolve = scope['resolve_controllers']
        for count in (1, 2):
            self.assertEqual(resolve(count, True, False, False, None), (count, count == 2))
            self.assertEqual(resolve(count, False, True, False, 'model.zip'), (count, False))
        self.assertEqual(resolve(None, False, False, True, None), (2, True))
        self.assertEqual(resolve(None, False, False, False, None), (1, False))
        for values in [(1, False, False, True, None), (2, True, False, False, 'm'),
                       (1, False, True, False, None), (2, False, True, True, 'm')]:
            with self.assertRaises(ValueError):
                resolve(*values)
