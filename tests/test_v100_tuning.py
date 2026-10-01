import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v100_tuning", ROOT / "tools/v100_tuning.py")
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)

class RuntimeRestoreTest(unittest.TestCase):
    def test_restoring_workers_preserves_later_usage_and_vision_settings(self):
        cfg = {"args": ["--vision", "--spec", "4"], "vision": {"gpu": True}}
        before = tuning.current(cfg, "workers")
        tuning.assign(cfg, "workers", "8")
        tuning.assign(cfg, "usage", "/tmp/usage")
        tuning.assign(cfg, "workers", before)
        self.assertEqual(cfg["args"], ["--vision", "--spec", "4"])
        self.assertEqual(cfg["env"]["STRATA_EXPERT_USAGE_DIR"], "/tmp/usage")
        self.assertTrue(cfg["vision"]["gpu"])
    def test_restore_existing_values(self):
        cfg = {"args": ["--pool-workers", "16"], "env": {"STRATA_EXPERT_USAGE_DIR": "old"}}
        workers = tuning.current(cfg, "workers")
        usage = tuning.current(cfg, "usage")
        tuning.assign(cfg, "workers", "8")
        tuning.assign(cfg, "usage", "new")
        tuning.assign(cfg, "workers", workers)
        tuning.assign(cfg, "usage", usage)
        self.assertEqual(cfg["args"], ["--pool-workers", "16"])
        self.assertEqual(cfg["env"]["STRATA_EXPERT_USAGE_DIR"], "old")
    def test_reject_duplicate_worker_arguments(self):
        with self.assertRaises(ValueError):
            tuning.current({"args": ["--pool-workers", "8", "--pool-workers", "16"]}, "workers")

if __name__ == "__main__":
    unittest.main()
