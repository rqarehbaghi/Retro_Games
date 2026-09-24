"""Backend selection must match the billing choice exposed by the CLI."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import writer  # noqa: E402


class WriterBackendTests(unittest.TestCase):
    def test_pinned_claude_code_does_not_fall_through(self):
        with mock.patch.dict(os.environ, {
                "ANTHROPIC_API_KEY": "configured",
                "GEMINI_API_KEY": "configured",
        }, clear=False), \
                mock.patch.object(writer, "claude_code_available", return_value=True), \
                mock.patch.object(writer, "generate_claude_code", return_value=None), \
                mock.patch.object(writer, "generate_claude") as anthropic, \
                mock.patch.object(writer, "generate_gemini") as gemini, \
                mock.patch.object(writer, "generate") as ollama:
            self.assertIsNone(writer.write("prompt", {}, backend="claude-code",
                                            verbose=False))
            anthropic.assert_not_called()
            gemini.assert_not_called()
            ollama.assert_not_called()

    def test_pinned_gemini_does_not_fall_through_to_ollama(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "configured"}, clear=False), \
                mock.patch.object(writer, "generate_gemini", return_value=None), \
                mock.patch.object(writer, "generate") as ollama:
            self.assertIsNone(writer.write("prompt", {}, backend="gemini",
                                            verbose=False))
            ollama.assert_not_called()

    def test_auto_still_cascades(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(writer, "claude_code_available", return_value=False), \
                mock.patch.object(writer, "generate", return_value={"ok": True}) as ollama:
            self.assertEqual(writer.write("prompt", {}, backend="auto", verbose=False),
                             {"ok": True})
            ollama.assert_called_once()


if __name__ == "__main__":
    unittest.main()
