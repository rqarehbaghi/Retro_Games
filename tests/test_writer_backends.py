"""Writer selection must never create a separately billed API path."""
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import writer  # noqa: E402


class WriterBackendTests(unittest.TestCase):
    def test_speech_keeps_link_words_but_never_reads_web_addresses(self):
        text = ("The original [Tetris came from Alexey Pajitnov in 1984]"
                "(https://www.tetris.com/about). More at https://example.com/x.")
        cleaned = writer.clean_for_speech(text)
        self.assertIn("Tetris came from Alexey Pajitnov in 1984", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("www.", cleaned)

    def test_only_subscription_and_local_backends_exist(self):
        self.assertEqual(writer.BACKENDS, ("chatgpt", "ollama", "auto"))
        self.assertEqual(writer.DEFAULT_BACKEND, "chatgpt")
        for removed in ("openai", "claude", "claude-code", "gemini"):
            self.assertNotIn(removed, writer.BACKENDS)

    def test_pinned_chatgpt_never_falls_through(self):
        with mock.patch.object(writer, "backend_available", return_value=True), \
                mock.patch.object(writer, "generate_chatgpt", return_value=None), \
                mock.patch.object(writer, "generate") as ollama:
            self.assertIsNone(writer.write("prompt", {}, backend="chatgpt",
                                            verbose=False))
            ollama.assert_not_called()

    def test_pinned_ollama_never_switches(self):
        with mock.patch.object(writer, "backend_available", return_value=True), \
                mock.patch.object(writer, "generate", return_value=None), \
                mock.patch.object(writer, "generate_chatgpt") as chatgpt:
            self.assertIsNone(writer.write("prompt", {}, backend="ollama",
                                            verbose=False))
            chatgpt.assert_not_called()

    def test_auto_cascades_from_chatgpt_to_ollama(self):
        with mock.patch.object(writer, "backend_available", return_value=True), \
                mock.patch.object(writer, "_write_one",
                                  side_effect=[None, {"ok": True}]) as run:
            self.assertEqual(writer.write("prompt", {}, backend="auto", verbose=False),
                             {"ok": True})
            self.assertEqual([call.args[0] for call in run.call_args_list],
                             ["chatgpt", "ollama"])

    def test_configured_auto_order_is_validated(self):
        self.assertEqual(writer.cascade_order(), ["chatgpt", "ollama"])
        self.assertEqual(writer.cascade_order("ollama,chatgpt"),
                         ["ollama", "chatgpt"])
        with self.assertRaises(ValueError):
            writer.cascade_order("chatgpt,openai")
        with self.assertRaises(ValueError):
            writer.cascade_order("chatgpt,chatgpt")

    def test_chatgpt_availability_requires_login_status(self):
        ok = subprocess.CompletedProcess([], 0, "Logged in using ChatGPT", "")
        with mock.patch.object(writer, "find_cli", return_value="/bin/codex"), \
                mock.patch("writer.subprocess.run", return_value=ok) as run:
            self.assertTrue(writer.chatgpt_available())
        self.assertEqual(run.call_args.args[0], ["/bin/codex", "login", "status"])
        with mock.patch.object(writer, "find_cli", return_value=None):
            self.assertFalse(writer.chatgpt_available())

    def test_chatgpt_generation_is_ephemeral_read_only_and_structured(self):
        def execute(cmd, **kwargs):
            answer = cmd[cmd.index("-o") + 1]
            with open(answer, "w", encoding="utf-8") as handle:
                json.dump({"ok": True}, handle)
            self.assertIn("--ephemeral", cmd)
            self.assertIn("--ignore-user-config", cmd)
            self.assertEqual(cmd[cmd.index("-s") + 1], "read-only")
            self.assertIn("--output-schema", cmd)
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
            self.assertNotEqual(os.path.abspath(kwargs["cwd"]),
                                os.path.dirname(os.path.dirname(__file__)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(writer, "find_cli", return_value="/bin/codex"), \
                mock.patch("writer.subprocess.run", side_effect=execute):
            self.assertEqual(writer.generate_chatgpt(
                "prompt", {"type": "object"}, verbose=False), {"ok": True})

    def test_chatgpt_malformed_output_fails_closed(self):
        def execute(cmd, **_kwargs):
            answer = cmd[cmd.index("-o") + 1]
            with open(answer, "w", encoding="utf-8") as handle:
                handle.write("not json")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(writer, "find_cli", return_value="/bin/codex"), \
                mock.patch("writer.subprocess.run", side_effect=execute):
            self.assertIsNone(writer.generate_chatgpt(
                "prompt", {"type": "object"}, verbose=False))


if __name__ == "__main__":
    unittest.main()
