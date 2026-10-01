"""Unified model selection, existing-config startup and process scoping; no model downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import setup
from tools.stop_server import ROOT, server_config


class UnifiedLauncher(unittest.TestCase):
    def test_shards(self):
        for family, model, count in (("orca", "IQ4_XS", 3), ("orca", "IQ3_XXS", 2), ("qwen", "IQ3_XXS", 2)):
            shards = setup.model_shards(family, model, "/models")
            self.assertEqual(len(shards), count)
            self.assertTrue(shards[-1].name.endswith(f"0000{count}-of-0000{count}.gguf"))

    def test_old_orca_config_can_be_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "strata-orca-iq4_xs.json"
            cfg.write_text(json.dumps({"args": ["--max-context", "262144"]}))
            choices = setup.choices_from_config(cfg)
            self.assertEqual((choices["family"], choices["model"], choices["context"]), ("orca", "IQ4_XS", 262144))

    def test_existing_config_starts_without_preparation(self):
        selected = setup.ROOT / "strata-orca-iq4_xs.json"
        with mock.patch.dict(os.environ, {"STRATA_UNIFIED_LAUNCHER": "1"}), \
             mock.patch.object(sys, "argv", ["setup.py", "--family", "orca", "--model", "IQ4_XS", "--yes"]), \
             mock.patch.object(setup, "data_folder", return_value=(Path('/tmp/data'), [])), \
             mock.patch.object(setup, "installed_configs", return_value=[selected]), \
             mock.patch.object(setup, "update_installed_engine"), \
             mock.patch.object(setup, "start", return_value=0) as start, \
             mock.patch.object(setup, "download") as download, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.main(), 0)
            self.assertEqual(start.call_args.args[0], selected)
            download.assert_not_called()

    def test_prepare_flag_does_not_start_server(self):
        selected = setup.ROOT / "strata-orca-iq4_xs.json"
        with mock.patch.dict(os.environ, {"STRATA_UNIFIED_LAUNCHER": "1"}), \
             mock.patch.object(sys, "argv", ["setup.py", "--family", "orca", "--model", "IQ4_XS", "--prepare-vision", "gpu", "--yes"]), \
             mock.patch.object(setup, "data_folder", return_value=(Path("/tmp/data"), [])), \
             mock.patch.object(setup, "installed_configs", return_value=[selected]), \
             mock.patch.object(setup, "configure_existing_vision") as prepare, \
             mock.patch.object(setup, "start") as start, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.main(), 0)
            self.assertTrue(prepare.call_args.kwargs["prepare_only"])
            start.assert_not_called()

    def test_prepare_vision_preserves_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "strata-orca-iq4_xs.json"
            original = {"args": ["--native", "/model.gguf", "--max-context", "262144"], "gpu": 0}
            cfg.write_text(json.dumps(original))
            with mock.patch.object(setup, "download"), \
                 mock.patch.object(setup, "gpus", return_value=[{"index": 0, "arch": "70"}]), \
                 mock.patch.object(setup, "get_llama_cpp", return_value=Path("/llama")), \
                 mock.patch.object(setup, "build_engine", return_value=Path("/engine")), \
                 contextlib.redirect_stdout(io.StringIO()):
                setup.configure_existing_vision(cfg, "orca", "gpu", td, True, prepare_only=True)
                self.assertEqual(json.loads(cfg.read_text()), original)
                setup.configure_existing_vision(cfg, "orca", "gpu", td, True)
                enabled = json.loads(cfg.read_text())
                self.assertTrue(enabled["vision"]["gpu"])
                self.assertIn("--vision", enabled["args"])
                self.assertEqual(enabled["args"][enabled["args"].index("--max-context") + 1], "262144")
                setup.configure_existing_vision(cfg, "orca", "none", td, True)
                self.assertEqual(json.loads(cfg.read_text()), original)

    def test_shell_routes_models_and_forwards_options(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            launcher = root / 'run-v100.sh'
            launcher.write_text((ROOT / 'run-v100.sh').read_text())
            helper = root / 'setup.sh'
            helper.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            helper.chmod(0o755)
            for name, family, model in [('orca', 'orca', 'IQ3_XXS'), ('orca-iq4_xs', 'orca', 'IQ4_XS'), ('coder', 'coder', 'IQ1_M'), ('IQ3_S', 'qwen', 'IQ3_S'), ('swift-iq2_xs', 'swift', 'IQ2_XS'), ('swift-iq3_xxs', 'swift', 'IQ3_XXS')]:
                result = subprocess.run(['sh', str(launcher), name, '--no-start'], capture_output=True, text=True, check=True)
                args = result.stdout.splitlines()
                self.assertEqual(args[args.index('--family') + 1], family)
                self.assertEqual(args[args.index('--model') + 1], model)
                self.assertEqual(args[-1], '--no-start')

    def test_stop_accepts_only_this_checkouts_servers(self):
        config = str(ROOT / 'strata-orca-iq4_xs.json')
        self.assertIsNotNone(server_config(['python', '-m', 'serve.server', '--config', config]))
        self.assertIsNotNone(server_config(['python', str(ROOT / 'serve/server.py'), '--config', config]))
        for argv in (['python', 'other.py', '--config', config], ['python', 'other.py', 'serve.server', '--config', config], ['python', '-m', 'serve.server', '--config', '/tmp/strata-orca.json'], ['python', '-m', 'serve.server', '--config']):
            self.assertIsNone(server_config(argv))


if __name__ == '__main__':
    unittest.main()
