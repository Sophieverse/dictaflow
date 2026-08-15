"""Config validation, atomicity and migration.

Most of these pin bugs that were found by attacking the running dashboard
rather than by reading the code, so they are written as "this exact hostile
input must be refused", not as abstract type checks.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from df import config  # noqa: E402


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self._patches = [
            mock.patch.object(config, "CONFIG_DIR", d),
            mock.patch.object(config, "CONFIG_FILE", d / "config.json"),
        ]
        for p in self._patches:
            p.start()
        self.dir = d

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()


class TestValidation(ConfigTestCase):
    def test_bool_is_not_accepted_where_an_int_is_expected(self):
        """`isinstance(True, int)` is True in Python.

        Without an explicit exclusion, POSTing {"preroll_ms": true} was
        accepted against the live server and set the pre-roll to 1ms. bool is
        a subclass of int, so every integer setting needs this.
        """
        self.assertFalse(config.validate("preroll_ms", True))
        self.assertFalse(config.validate("preroll_ms", False))
        self.assertFalse(config.validate("max_session_seconds", True))
        self.assertTrue(config.validate("preroll_ms", 500))

    def test_int_is_not_accepted_where_a_bool_is_expected(self):
        self.assertFalse(config.validate("streaming", 1))
        self.assertFalse(config.validate("streaming", "yes"))
        self.assertTrue(config.validate("streaming", True))

    def test_ranges_are_enforced(self):
        self.assertFalse(config.validate("preroll_ms", -1))
        self.assertFalse(config.validate("preroll_ms", 99999))
        self.assertFalse(config.validate("max_session_seconds", 5))
        self.assertTrue(config.validate("max_session_seconds", 1200))

    def test_enums_are_enforced(self):
        for good in ("none", "light", "medium", "high"):
            self.assertTrue(config.validate("cleanup_level", good))
        self.assertFalse(config.validate("cleanup_level", "aggressive"))
        self.assertFalse(config.validate("bar_dock", "top"))
        self.assertFalse(config.validate("paste_method", "magic"))

    def test_unknown_keys_are_refused(self):
        self.assertFalse(config.validate("not_a_real_key", 1))

    def test_bindings_must_name_real_keys_and_be_unique(self):
        self.assertTrue(config.validate(
            "bindings", {"turbo": "alt_r", "small": "cmd_r"}))
        # An unknown key name would make that slot silently dead — the key
        # would simply never fire, with no error anywhere.
        self.assertFalse(config.validate("bindings", {"turbo": "space"}))
        # Two slots on one key makes one of them unreachable.
        self.assertFalse(config.validate(
            "bindings", {"turbo": "alt_r", "small": "alt_r"}))
        self.assertFalse(config.validate("bindings", {"bogus_slot": "alt_r"}))
        self.assertFalse(config.validate("bindings", {}))

    def test_dictionary_and_snippet_shapes(self):
        self.assertTrue(config.validate("dictionary", [{"from": "a", "to": "b"}]))
        self.assertFalse(config.validate("dictionary", [{"from": "a"}]))
        self.assertFalse(config.validate("dictionary", [{"from": "", "to": "b"}]))
        self.assertFalse(config.validate("dictionary", [{"from": 1, "to": "b"}]))
        self.assertFalse(config.validate("dictionary", "not a list"))
        self.assertFalse(config.validate(
            "snippets", [{"trigger": "x", "text": "y" * 5000}]))
        self.assertTrue(config.validate(
            "snippets", [{"trigger": "x", "text": "y" * 100}]))

    def test_ollama_host_must_be_a_url(self):
        self.assertTrue(config.validate("ollama_host", "http://localhost:11434"))
        self.assertFalse(config.validate("ollama_host", "localhost:11434"))
        self.assertFalse(config.validate("ollama_host", "file:///etc/passwd"))


class TestUpdate(ConfigTestCase):
    def test_only_editable_keys_are_written_and_the_rest_reported(self):
        cfg, rejected = config.update({
            "language": "fr",
            "models": {"turbo": "attacker/pwned"},
            "groq_api_key": "STOLEN",
            "backend": "groq",
        })
        self.assertEqual(cfg["language"], "fr")
        self.assertEqual(cfg["models"], config.DEFAULTS["models"])
        self.assertEqual(cfg["groq_api_key"], "")
        self.assertEqual(cfg["backend"], "local")
        # Reported, not silently dropped: a settings form that appears to save
        # but doesn't is worse than one that says no.
        self.assertEqual(set(rejected), {"models", "groq_api_key", "backend"})

    def test_model_path_is_not_writable(self):
        """The escalation that made the old CSRF hole serious.

        `models` feeds mlx_whisper's path_or_hf_repo, which downloads and
        loads an arbitrary Hugging Face repo — and DictaFlow types its output
        as keystrokes into the focused app.
        """
        self.assertNotIn("models", config.EDITABLE)
        self.assertNotIn("backend", config.EDITABLE)
        self.assertNotIn("groq_api_key", config.EDITABLE)

    def test_refuses_to_write_over_an_unreadable_config(self):
        """A single corrupt byte used to turn Save into a wipe: the read fell
        back to defaults and the write then rebuilt the file from them,
        destroying the API keys."""
        config.CONFIG_FILE.write_text("{ this is not json")
        with self.assertRaises(RuntimeError):
            config.update({"language": "fr"})


class TestLoadSave(ConfigTestCase):
    def test_missing_file_returns_defaults(self):
        cfg = config.load()
        self.assertEqual(cfg["cleanup_level"], config.DEFAULTS["cleanup_level"])
        self.assertNotIn("_error", cfg)

    def test_corrupt_file_is_preserved_not_overwritten(self):
        config.CONFIG_FILE.write_text('{"language": ')
        cfg = config.load()
        self.assertIn("_error", cfg)
        self.assertTrue(config.CONFIG_FILE.with_suffix(".broken.json").exists(),
                        "the damaged file must be kept — it may hold the only "
                        "copy of an API key")

    def test_invalid_values_on_disk_are_ignored_not_fatal(self):
        config.CONFIG_FILE.write_text(json.dumps({
            "version": 2, "preroll_ms": True, "cleanup_level": "wild",
            "language": "en",
        }))
        cfg = config.load()
        self.assertEqual(cfg["preroll_ms"], config.DEFAULTS["preroll_ms"])
        self.assertEqual(cfg["cleanup_level"], config.DEFAULTS["cleanup_level"])
        self.assertEqual(cfg["language"], "en")

    def test_save_is_atomic(self):
        """A crash mid-write used to leave a truncated file, which the next
        read called corrupt, which the next save finalised into total loss."""
        config.save({**config.DEFAULTS, "language": "de"})
        leftovers = [p for p in self.dir.iterdir() if p.name.startswith(".config-")]
        self.assertEqual(leftovers, [], "temp file was not cleaned up")
        self.assertEqual(json.loads(config.CONFIG_FILE.read_text())["language"], "de")

    def test_private_keys_are_not_persisted(self):
        config.save({**config.DEFAULTS, "_error": "boom"})
        self.assertNotIn("_error", json.loads(config.CONFIG_FILE.read_text()))

    def test_concurrent_saves_never_produce_a_corrupt_file(self):
        def writer(n):
            for i in range(20):
                config.save({**config.DEFAULTS, "initial_prompt": f"{n}-{i}"})
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        cfg = config.load()
        self.assertNotIn("_error", cfg, "a concurrent save corrupted the file")


class TestMigration(ConfigTestCase):
    def test_v1_config_migrates(self):
        config.CONFIG_FILE.write_text(json.dumps({
            "backend": "local",
            "local_whisper_model": "/models/whisper-small-mlx",
            "cleanup_enabled": False,
            "cleanup_prompt": "old prompt",
            "cleanup_model": "llama",
            "language": "en",
            "groq_api_key": "keepme",
        }))
        cfg = config.load()
        self.assertEqual(cfg["version"], config.SCHEMA_VERSION)
        self.assertEqual(cfg["models"]["small"], "/models/whisper-small-mlx")
        self.assertEqual(cfg["cleanup_level"], "light")
        self.assertNotIn("cleanup_prompt", cfg)
        self.assertEqual(cfg["groq_api_key"], "keepme",
                         "migration must not lose the API key")

    def test_v1_with_cleanup_enabled_maps_to_medium(self):
        config.CONFIG_FILE.write_text(json.dumps({"cleanup_enabled": True}))
        self.assertEqual(config.load()["cleanup_level"], "medium")

    def test_migration_is_idempotent(self):
        config.CONFIG_FILE.write_text(json.dumps({
            "local_whisper_model": "/models/whisper-large-v3-turbo",
            "cleanup_enabled": True,
        }))
        first = config.load()
        config.save(first)
        second = config.load()
        self.assertEqual(first["models"], second["models"])
        self.assertEqual(first["cleanup_level"], second["cleanup_level"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
