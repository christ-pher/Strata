"""Benchmark vision comparisons load the encoder and preserve runtime settings."""
import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v100_bench", ROOT / "tools/v100_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

class BenchmarkConfigTest(unittest.TestCase):
    def setUp(self):
        self.source = {"exe": "engine/strata", "args": ["--vision", "--spec", "4"],
                       "env": {"STRATA_PREFILL_RING": "8"}, "vision": {"gpu": True}, "api_key": "private"}
    def test_text_only_removes_encoder_and_engine_flag(self):
        cfg = bench.benchmark_config(self.source, workers=8)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])
        self.assertEqual(cfg["args"][-2:], ["--pool-workers", "8"])
        self.assertNotIn("api_key", cfg)
        self.assertEqual(self.source["args"][0], "--vision")
    def test_gpu_vision_is_actually_loaded(self):
        cfg = bench.benchmark_config(self.source, vision="gpu", engine_env=["STRATA_BF16_REUSE_W=0"])
        self.assertEqual(cfg["vision"], {"gpu": True})
        self.assertIn("--vision", cfg["args"])
        self.assertEqual(cfg["env"], {"STRATA_PREFILL_RING": "8", "STRATA_BF16_REUSE_W": "0"})
        self.assertEqual(self.source["env"], {"STRATA_PREFILL_RING": "8"})

if __name__ == "__main__":
    unittest.main()
